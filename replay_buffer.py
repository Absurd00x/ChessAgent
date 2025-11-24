# replay_buffer.py
import random
import os
from typing import List, Tuple

import numpy as np

from constants import (
    TOTAL_LAYERS,
    TOTAL_MOVES,
    REPLAY_CAPACITY,
    DEFAULT_REPLAY_PATH
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

def save_replay_buffer(path: str = DEFAULT_REPLAY_PATH) -> None:
    """
    Сохраняем текущее содержимое буфера (только первые sz элементов)
    в .npz-файл. Используем сжатие, чтобы меньше занимать место.
    """
    rb = replay_buffer
    if rb.sz == 0:
        return

    np.savez_compressed(
        path,
        obs=rb.obs[:rb.sz],
        pi=rb.pi[:rb.sz],
        z=rb.z[:rb.sz],
        idx=np.int64(rb.idx),
        sz=np.int64(rb.sz),
        capacity=np.int64(rb.capacity),
    )


def load_replay_buffer(path: str = DEFAULT_REPLAY_PATH) -> bool:
    """
    Пытаемся загрузить буфер из файла.
    Возвращает True, если успешно, False, если файл не найден.
    """
    if not os.path.exists(path):
        return False

    data = np.load(path, allow_pickle=False)

    obs = data["obs"]
    pi = data["pi"]
    z = data["z"]
    saved_idx = int(data["idx"])
    saved_sz = int(data["sz"])
    saved_capacity = int(data["capacity"])

    rb = replay_buffer

    # Если сохранённый capacity больше текущего — пересоздадим массивы
    if saved_capacity > rb.capacity:
        rb.capacity = saved_capacity
        rb.obs = np.zeros((saved_capacity, TOTAL_LAYERS, 8, 8), dtype=np.float32)
        rb.pi = np.zeros((saved_capacity, TOTAL_MOVES), dtype=np.float32)
        rb.z = np.zeros((saved_capacity,), dtype=np.float32)

    # Копируем данные
    rb.obs[:saved_sz] = obs
    rb.pi[:saved_sz] = pi
    rb.z[:saved_sz] = z

    rb.sz = saved_sz
    rb.idx = saved_idx % rb.capacity

    return True
