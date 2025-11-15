import gymnasium as gym

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random
import torch.distributions as D

from itertools import count
from nw import CNNActorCritic
from env import TOTAL_MOVES


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
        self.legal_masks = torch.zeros((self.T, self.n_workers, TOTAL_MOVES), dtype=torch.bool, device=self.device)
        self.ptr = 0

    def add(self, s, a, r, d, logp, v, legal_mask):
        self.states[self.ptr].copy_(s)
        self.actions[self.ptr].copy_(a)
        self.rewards[self.ptr].copy_(r)
        self.dones[self.ptr].copy_(d)
        self.logpas[self.ptr].copy_(logp)
        self.values[self.ptr].copy_(v)
        self.legal_masks[self.ptr].copy_(legal_mask)
        self.ptr += 1

    @torch.no_grad()
    def compute_returns_adv(self, last_vaules):
        # GAE по (T, N)
        T, N = self.T, self.n_workers
        adv = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros(N, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            done = self.dones[t]
            if t == T - 1:
                next_value = last_vaules
            else:
                next_value = self.values[t + 1]
            next_value = (1.0 - done) * next_value
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
        legal_masks = self.legal_masks.reshape(T * N, TOTAL_MOVES)
        return states, actions, logpas, adv, ret, legal_masks


class PPO:
    def __init__(self, make_env_function):
        self.make_env_function = make_env_function
        self.rollout_len = 256
        self.n_workers = 8
        self.gamma = 0.99
        self.lam = 0.95

        self.lr = 1e-4
        # self.policy_coef = 1.0
        self.value_coef = 0.5
        self.entropy_coef = 1e-3
        # суть PPO
        self.policy_clip = 0.2
        self.value_clip = 0.2

        self.epochs = 4
        self.batch_ratio = 0.25

        self.eval_games = 300        # сколько партий в evaluate
        self.eval_interval = 3000    # раз в сколько эпизодов вызывать evaluate

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

    @torch.no_grad()
    def _evaluate(self, evaluate_episodes=300):
        venv = gym.vector.SyncVectorEnv([self.make_env_function for _ in range(self.n_workers)])
        cur_obs, infos = venv.reset()

        episodes = evaluate_episodes
        finished_episodes = 0
        pbar = tqdm.trange(episodes, desc="evaluating...")
        wins = 0
        illegal = 0
        captures = 0

        while finished_episodes < episodes:
            obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
            legal_mask = torch.as_tensor(infos["legal_mask"], device=self.device, dtype=torch.bool)

            actions = self.model.make_move(obs, legal_mask)
            actions = actions.cpu().numpy()

            next_obs, rewards, terms, truncs, infos = venv.step(actions)
            dones = np.logical_or(terms, truncs)
            done_this_time = 0

            for i, d in enumerate(dones):
                illegal += (not infos["legal_move"][i])
                if d:
                    done_this_time += 1
                    captures += venv.envs[i].my_captures

                    if infos["winner"][i] is True:
                        wins += 1

                    obs_i, info_i = venv.envs[i].reset()
                    next_obs[i] = obs_i
                    infos["legal_mask"][i] = info_i["legal_mask"]
            cur_obs = next_obs

            pbar.update(done_this_time)
            finished_episodes += done_this_time

        venv.close()


        winrate = wins / finished_episodes * 100 if finished_episodes > 0 else 0.0
        print(f"[EVAL] Games: {finished_episodes}")
        print(f"[EVAL] Winrate (greedy): {winrate:.8f}%")
        print(f"[EVAL] illegal: {illegal}")
        print(f"[EVAL] captures: {captures}")

    def train(self, desired_winrate = 80, seed=42, plot=False):
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        venv = gym.vector.SyncVectorEnv([self.make_env_function for _ in range(self.n_workers)])
        cur_obs, infos = venv.reset(seed=seed)
        episode = 1

        wins = 0
        total = 0

        scores = []

        last_episode_shown = 0
        last_evaluation = 0

        print("Started training")
        while total < 500 or (total > 0 and wins / total * 100 < desired_winrate):
            if episode // 100 != last_episode_shown:
                print(f"Episodes: {episode}")
                last_episode_shown = episode // 100
            if episode - last_evaluation >= self.eval_interval:
                self._evaluate(self.eval_games)
                last_evaluation = episode
            # Collect rollout
            self.buffer.reset()
            for t in range(self.rollout_len):
                obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
                legal_mask = torch.as_tensor(infos["legal_mask"], device=self.device, dtype=torch.bool)
                with torch.no_grad():
                    logits, values = self.model(obs)
                    logits[~legal_mask] = -1e9
                    dist = D.Categorical(logits=logits)
                    actions = dist.sample()
                    logpas = dist.log_prob(actions)

                actions_np = actions.cpu().numpy()
                next_obs, rewards, terms, truncs, infos = venv.step(actions_np)
                dones = np.logical_or(terms, truncs).astype(np.float32)

                self.buffer.add(
                    obs,
                    actions,
                    torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                    torch.as_tensor(dones, dtype=torch.float32, device=self.device),
                    logpas,
                    values,
                    legal_mask
                )

                # Если эпизод закончился, то ресетим эту среду
                for i, d in enumerate(dones):
                    if d:
                        total += 1
                        episode += 1
                        obs_i, info_i = venv.envs[i].reset()
                        next_obs[i] = obs_i
                        infos["legal_mask"][i] = info_i["legal_mask"]
                cur_obs = next_obs
            last_obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_values = self.model.values_only(last_obs)
            states, actions, old_logpas, adv, ret, legal_masks = self.buffer.compute_returns_adv(last_values)
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
                    lm_b = legal_masks[mb_idx]

                    logits, values = self.model(s_b)
                    logits[~lm_b] = -1e9
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


