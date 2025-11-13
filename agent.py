import gymnasium as gym

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random

from itertools import count
from nw import CNNActorCritic
from env import make_env_function

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
    def __init__(self):
        self.rollout_len = 256
        self.n_workers = 8
        self.gamma = 0.99
        self.lam = 0.95

        self.lr = 3e-4
        # self.policy_coef = 1.0
        self.value_coef = 0.5
        self.entropy_coef = 1e-3
        # суть PPO
        self.policy_clip = 0.2
        self.value_clip = 0.2

        self.epochs = 4
        self.batch_ratio = 0.25

        env = make_env_function()
        obs_shape = env.observation_space.shape
        env.close()

        self.device = "cuda:0"

        self.model = CNNActorCritic()

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

    def train(self, desired_winrate = 60, seed=42, plot=False):
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        venv = gym.vector.SyncVectorEnv([make_env_function for _ in range(self.n_workers)])
        cur_obs, infos = venv.reset(seed=seed)
        episode = 1

        wins = 0
        total = 0

        scores = []

        last = 0
        illegal = 0

        print("Started training")
        while total < 500 or (total > 0 and wins / total * 100 < desired_winrate):
            if episode // 100 != last:
                print(f"Episodes: {episode}")
                print(f"Winrate: {wins/total * 100:.8f}%")
                print(f"illegal: {illegal}")
                illegal = 0
                last = episode // 100
            # Collect rollout
            self.buffer.reset()
            for t in range(self.rollout_len):
                obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
                legal_mask = torch.as_tensor(infos["legal_mask"], device=self.device, dtype=torch.bool)
                with torch.no_grad():
                    actions, logpas, _, values = self.model.full_pass(obs, legal_mask)
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
                    illegal += (rewards[i] == -0.1)
                    if d:
                        if rewards[i] > 0:
                            wins += 1
                            if plot:
                                scores.append(1.0)
                        else:
                            scores.append(0.0)
                        total += 1
                        episode += 1

                        obs_i, info_i = venv.envs[i].reset()
                        next_obs[i] = obs_i
                cur_obs = next_obs
            last_obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_values = self.model.values_only(last_obs)
            states, actions, old_logpas, adv, ret = self.buffer.compute_returns_adv(last_values)
            # optimize
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            n = states.size(0)
            batch_size = max(1, int(self.batch_ratio * n))

            # policy update
            for _ in range(self.epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, batch_size):
                    end = start + batch_size
                    mb_idx = idx[start:end]

                    # state batch = states[minibatch index]
                    s_b = states[mb_idx]
                    a_b = actions[mb_idx]
                    adv_b = adv[mb_idx]
                    old_lp_b = old_logpas[mb_idx]
                    ret_b = ret[mb_idx]

                    logits, values = self.model(s_b)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(a_b)
                    entropy = dist.entropy().mean()

                    # policy loss
                    ratio = (new_logp - old_lp_b).exp()
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio,
                                        1.0 - self.policy_clip,
                                        1.0 + self.policy_clip) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                    values = values.squeeze(-1)
                    v_old = values.detach()
                    v_clipped = v_old + (values - v_old).clamp(-self.value_clip, self.value_clip)
                    v_loss1 = (values - ret_b).pow(2)
                    v_loss2 = (v_clipped - ret_b).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                    # entropy regularization
                    entropy_loss = -entropy

                    # общий loss
                    loss = (policy_loss
                            + self.value_coef * value_loss
                            + self.entropy_coef * entropy_loss)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    opt.step()

        venv.close()
        if plot:
            wins = 0
            total = 0
            ys = []
            for i in range(min(100, len(scores))):
                wins += scores[i]
                total += 1
                ys.append(wins / total)
            for i in range(100, len(scores)):
                wins += scores[i]
                wins -= scores[i - 100]
                ys.append(wins / total)
            xs = np.arange(1, len(ys) + 1)
            tail_scores = np.asarray(scores[:len(ys)], dtype=float)

            plt.figure()
            plt.plot(xs, tail_scores, linewidth=1, alpha=0.35, label='score (win=1 / loss=0)')
            plt.plot(xs, ys, linewidth=2, label='rolling mean (≤100 games)')
            plt.ylim(-0.05, 1.05)
            plt.xlabel('Game #')
            plt.ylabel('Win (1) / Loss (0)')
            plt.title('Rolling winrate')
            plt.legend()
            plt.tight_layout()
            plt.show()


