# replay_buffer.py
import random
from typing import List, Tuple

import numpy as np

from constants import (
    TOTAL_LAYERS,
    TOTAL_MOVES,
    REPLAY_CAPACITY
)


class ReplayBuffer:
    """
    Плотный replay buffer на предвыделенных numpy-массивах.
    Хранит (obs, pi, z):
      obs: [TOTAL_LAYERS, 8, 8]
      pi : [TOTAL_MOVES]
      z  : scalar float
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

        # Основные массивы
        self.obs = np.zeros(
            (capacity, TOTAL_LAYERS, 8, 8), dtype=np.float32
        )
        self.pi = np.zeros(
            (capacity, TOTAL_MOVES), dtype=np.float32
        )
        self.z = np.zeros(
            (capacity,), dtype=np.float32
        )

        # idx — позиция следующей записи (кольцевой буфер)
        # sz  — фактический размер (сколько уже реально занято)
        self.idx = 0
        self.sz = 0

    def __len__(self) -> int:
        return self.sz

    def add_many(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        data — список (obs, pi_vec, z) из self_play_game.
        """
        for obs, pi_vec, z in data:
            # На всякий случай приводим к float32
            obs = obs.astype(np.float32, copy=False)
            pi_vec = pi_vec.astype(np.float32, copy=False)
            z = np.float32(z)

            self.obs[self.idx] = obs
            self.pi[self.idx] = pi_vec
            self.z[self.idx] = z

            self.idx = (self.idx + 1) % self.capacity
            if self.sz < self.capacity:
                self.sz += 1

    def sample(self, batch_size: int):
        """
        Возвращает (obs_batch, pi_batch, z_batch) как np.ndarray.
        """
        if self.sz == 0:
            raise ValueError("ReplayBuffer пуст, нечего sample'ить")

        batch_size = min(batch_size, self.sz)
        # равномерно выбираем индексы из [0, sz)
        indices = np.random.choice(self.sz, size=batch_size, replace=False)

        obs_batch = self.obs[indices]
        pi_batch = self.pi[indices]
        z_batch = self.z[indices]

        return obs_batch, pi_batch, z_batch


# Глобальный буфер, который импортируем в agent.py
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
