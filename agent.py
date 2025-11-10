import gymnasium as gym

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random

from itertools import count
from env import ChessEnv

class RolloutBuffer:
    def __init__(self, obs_shape, n_workers, rollout_len, gamma, lam, device):
        self.obs_shape = obs_shape
        self.n_workers = n_workers
        self.T = rollout_len
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.reset()

    def reset(self):
        self.states = torch.zeros((self.T, self.n_workers, *self.obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.T, self.n_workers), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((self.T, self.n_workers), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.T, self.n_workers), dtype=torch.float32, device=self.device)
        self.logpas = torch.zeros((self.T, self.n_workers), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.T, self.n_workers), dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add(self, s, a, r, d, logp, v):
        self.states[self.ptr].copy_(s)
        self.actions[self.ptr].copy_(a)
        self.rewards[self.ptr].copy_(r)
        self.dones[self.ptr].copy_(d)
        self.logpas[self.ptr].copy_(logp)
        self.values[self.ptr].copy_(v)
        self.ptr += 1

    @torch.no_grad()
    def compute_returns_adv(self, last_vaules):
        # GAE по (T, N)
        T, N = self.T, self.n_workers
        adv = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros(N, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            done = self.dones[t]
            next_value = (1.0 - done) * last_vaules if t == T-1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            last_gae = delta + self.gamma * self.lam * (1.0 - done) * last_gae
            adv[t] = last_gae
        ret = adv + self.values

        # Выпрямляем
        states = self.states.reshape(T * N, *self.obs_shape)
        actions = self.actions.reshape(T * N)
        logpas = self.logpas.reshape(T * N)
        adv = adv.reshape(T * N)
        ret = ret.reshape(T * N)
        return states, actions, logpas, adv, ret


class PPO:
    def __init__(self,
                 PolicyModel,
                 ValueModel):
        self.rollout_len = 256
        self.n_workers = 8
        self.gamma = 0.99
        self.lam = 0.95
        self.policy_lr = 3e-4
        self.value_lr = 1e-3
        # суть PPO
        self.policy_clip = 0.2
        self.value_clip = 0.2

        self.policy_epochs = 4
        self.value_epochs = 4
        self.batch_ratio = 0.25
        self.entropy_weight=1e-3

        env = ChessEnv()
        obs_shape = env.observation_space.shape
        nA = env.action_space.n
        env.close()

        self.device = "cuda:0"

        self.actor = PolicyModel(obs_shape[0], nA)
        self.critic = ValueModel(obs_shape[0])

        self.buffer = RolloutBuffer(obs_shape,
                                    self.n_workers,
                                    self.rollout_len,
                                    self.gamma,
                                    self.lam,
                                    self.device)

    def _state_to_device(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def train(self, episodes=10000, seed=42, demo_last=True):
        self.actor.to(self.device)
        self.critic.to(self.device)
        policy_opt = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)
        value_opt = torch.optim.Adam(self.critic.parameters(), lr=self.value_lr)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        def make_one():
            return ChessEnv()

        venv = gym.vector.SyncVectorEnv([make_one for _ in range(self.n_workers)])
        cur_obs, _ = venv.reset(seed=seed)
        episode_rewards = []
        finished_episodes = 0

        pbar = tqdm.trange(episodes, desc="PPO episodes")
        while finished_episodes < episodes:
            if finished_episodes % 500 == 0:
                self.evaluate(n_episodes=1, record=True)
            # Collect rollout
            self.buffer.reset()
            for t in range(self.rollout_len):
                obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    actions, logpas, _ = self.actor.full_pass(obs)
                    values = self.critic(obs)
                actions_np = actions.cpu().numpy()
                next_obs, rewards, terms, truncs, infos = venv.step(actions_np)
                dones = np.logical_or(terms, truncs).astype(np.float32)

                self.buffer.add(
                    obs,
                    actions,
                    torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                    torch.as_tensor(dones, dtype=torch.float32, device=self.device),
                    logpas,
                    values
                )

                # Если эпизод закончился, то ресетим эту среду
                for i, d in enumerate(dones):
                    if d:
                        obs_i, info_i = venv.envs[i].reset()
                        next_obs[i] = obs_i
                cur_obs = next_obs
            last_obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_values = self.critic(last_obs)
            states, actions, old_logpas, adv, ret = self.buffer.compute_returns_adv(last_values)
            # optimize
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            n = states.size(0)
            batch_size = max(1, int(self.batch_ratio * n))

            # policy update
            for _ in range(self.policy_epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, batch_size):
                    end = start + batch_size
                    mb_idx = idx[start:end]

                    # state batch = states[minibatch index]
                    s_b = states[mb_idx]
                    a_b = actions[mb_idx]
                    adv_b = adv[mb_idx]
                    old_lp_b = old_logpas[mb_idx]

                    logits = self.actor(s_b)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(a_b)
                    entropy = dist.entropy().mean()

                    ratio = (new_logp - old_lp_b).exp()
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio,
                                        1.0 - self.policy_clip,
                                        1.0 + self.policy_clip) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -self.entropy_weight * entropy

                    policy_opt.zero_grad()
                    (policy_loss + entropy_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    policy_opt.step()
            # value update
            for _ in range(self.value_epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, batch_size):
                    end = start + batch_size
                    mb_idx = idx[start:end]

                    s_b = states[mb_idx]
                    ret_b = ret[mb_idx]

                    v_pred = self.critic(s_b)
                    v_old = v_pred.detach()

                    v_clipped = v_old + (v_pred - v_old).clamp(-self.value_clip, self.value_clip)
                    loss1 = (v_pred - ret_b).pow(2)
                    loss2 = (v_clipped - ret_b).pow(2)
                    v_loss = 0.5 * torch.max(loss1, loss2).mean()

                    value_opt.zero_grad()
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    value_opt.step()
            #statistics
            eval_ret, _ = self.evaluate(n_episodes=5, record=False)
            episode_rewards.append(eval_ret)
            pbar.update(1)
            finished_episodes += 1

        venv.close()
        rewards_plot = np.asarray(episode_rewards, dtype=float)
        episodes = np.arange(1, len(rewards_plot) + 1)

        plt.figure()
        plt.plot(episodes, rewards_plot, alpha=0.4, label="episode return")

        # (опционально) сглаживание скользящим средним
        if len(rewards_plot) >= 20:
            w = 20
            ma = np.convolve(rewards_plot, np.ones(w) / w, mode="valid")
            plt.plot(np.arange(w, len(rewards_plot) + 1), ma, linewidth=2.0, label=f"MAE{w}")

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training returns")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("Training complete. Evaluating...")
        final_eval_score, score_std = self.evaluate(n_episodes=10, record=False)
        print(f'Final evaluation score {final_eval_score:.2f}\u00B1{score_std:.2f}')
        if demo_last:
            self.evaluate(n_episodes=2, record=True)


    def evaluate(self, n_episodes=1, record=False):
        env = ChessEnv()
        rewards = []
        try:
            for _ in range(n_episodes):
                s, info = env.reset()
                ep_ret = 0e0
                for _ in count():
                    s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        a = self.actor.greedy_action(s_t).item()
                    s, r, terminated, truncated, info = env.step(a)
                    ep_ret += r
                    if terminated or truncated:
                        break
                rewards.append(ep_ret)
        finally:
            env.close()
        return float(np.mean(rewards)), float(np.std(rewards))
