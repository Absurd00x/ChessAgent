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
    Если в файле записей больше, чем текущий capacity,
    самые старые записи отбрасываются, остаются только
    последние `replay_buffer.capacity` позиций.
    Возвращает True, если успешно, False, если файл не найден.
    """
    if not os.path.exists(path):
        return False

    data = np.load(path, allow_pickle=False)

    obs = data["obs"]  # shape: (saved_sz, TOTAL_LAYERS, 8, 8)
    pi = data["pi"]    # shape: (saved_sz, TOTAL_MOVES)
    z = data["z"]      # shape: (saved_sz,)
    saved_idx = int(data["idx"])
    saved_sz = int(data["sz"])
    saved_capacity = int(data["capacity"])

    rb = replay_buffer
    new_capacity = rb.capacity  # что задано текущей константой REPLAY_CAPACITY

    if saved_sz == 0:
        # В файле пустой буфер
        rb.sz = 0
        rb.idx = 0
        return True

    # Сколько реально хотим сохранить после загрузки
    keep_sz = min(saved_sz, new_capacity)

    # Массивы под новый буфер (ориентируемся на ТЕКУЩИЙ capacity)
    rb.obs = np.zeros((new_capacity, TOTAL_LAYERS, 8, 8), dtype=np.float32)
    rb.pi  = np.zeros((new_capacity, TOTAL_MOVES), dtype=np.float32)
    rb.z   = np.zeros((new_capacity,), dtype=np.float32)
    rb.capacity = new_capacity

    if saved_sz <= new_capacity:
        # Помещаются все записи — просто копируем "как есть"
        rb.obs[:saved_sz] = obs[:saved_sz]
        rb.pi[:saved_sz]  = pi[:saved_sz]
        rb.z[:saved_sz]   = z[:saved_sz]

        rb.sz = saved_sz
        # Буфер не полный => следующий индекс — просто sz
        rb.idx = saved_sz % new_capacity
    else:
        # saved_sz > new_capacity — нужно выбросить самые старые
        # Оставляем последние `keep_sz` записей по времени

        if saved_sz < saved_capacity:
            # Буфер не успел заполниться при сохранении:
            # хронологический порядок — просто индексы 0..saved_sz-1
            start = saved_sz - keep_sz
            sel = np.arange(start, saved_sz)
        else:
            # Буфер был полон, хронологический порядок — кольцевой.
            # saved_idx — позиция СЛЕДУЮЩЕЙ записи;
            # самая старая запись сейчас в индексе saved_idx.
            # Хронологический порядок индексов:
            # (saved_idx + 0) % saved_capacity, ...,
            # (saved_idx + saved_sz - 1) % saved_capacity.
            start_ch = saved_sz - keep_sz  # с какого места брать последние keep_sz
            j = np.arange(start_ch, saved_sz)  # хронологические индексы
            sel = (saved_idx + j) % saved_capacity

        # Перекладываем выбранные индексы плотно с 0 до keep_sz-1
        rb.obs[:keep_sz] = obs[sel]
        rb.pi[:keep_sz]  = pi[sel]
        rb.z[:keep_sz]   = z[sel]

        rb.sz = keep_sz
        # Если буфер теперь полон, idx = 0 (след. запись перезатрет самую старую),
        # если нет — idx = sz.
        rb.idx = keep_sz % new_capacity

    return True
