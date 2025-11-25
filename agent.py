import math
import random
import numpy as np
from env import board_to_planes, board_to_obs, action_id_to_move, move_to_id
import chess
import torch
import torch.nn.functional as F
from nw import CNNActorCritic
from collections import deque
from constants import (
    LAST_POSITIONS,
    TOTAL_MOVES,
    PIECE_VALUES,
    BATCH_SIZE,
    MIN_REPLAY_SIZE,
    TRAIN_STEPS_PER_ITER,
)
from replay_buffer import replay_buffer

position_deque = deque(maxlen=LAST_POSITIONS)

def policy_to_pi_vector(board: chess.Board, policy:dict) -> np.ndarray:
    """
    Преобразуем policy в виде {chess.Move: prob} в вектор длины TOTAL_MOVES.
    """

    pi = np.zeros(TOTAL_MOVES, dtype=np.float32)
    for move, prob in policy.items():
        action_id = move_to_id(board, move)
        assert 0 <= action_id < TOTAL_MOVES, "encode_action вернул некорректный id"
        pi[action_id] = prob
    return pi


def self_play_game(model: CNNActorCritic,
                   num_simulations=64,
                   max_moves=200,
                   device="cpu"):
    """
    Играем партию MCTS vs MCTS.
    Возвращаем список (obs, pi_vec, z) для каждого хода.
      obs   : np.ndarray [TOTAL_LAYERS, 8, 8]
      pi_vec: np.ndarray [TOTAL_MOVES]
      z     : скаляр в {-1, 0, +1} с точки зрения игрока, который делал ход.
    """
    mcts = MCTS(number_of_simulations=num_simulations,
                model=model,
                device=device)
    board = chess.Board()
    position_deque.clear()

    trajectory = []
    moves_cnt = 0

    temperature_moves = 20  # первые 20 полуходов выбираем с температурой

    while not board.is_game_over() and moves_cnt < max_moves:
        player = board.turn
        obs = board_to_obs(board, position_deque)

        # MCTS c шумом Дирихле в корне
        move, policy = mcts.run(board, add_dirichlet_noise=True)

        # π всегда берём как нормализованные visit counts
        pi_vec = policy_to_pi_vector(board, policy)

        # Для первых ходов выбираем действие случайно из policy
        if moves_cnt < temperature_moves:
            moves_list = list(policy.keys())
            probs = np.array([policy[m] for m in moves_list], dtype=np.float64)
            s = probs.sum()
            if s <= 0:
                # деградация на всякий случай
                probs = np.ones(len(moves_list), dtype=np.float64) / len(moves_list)
            else:
                probs /= s
            idx = np.random.choice(len(moves_list), p=probs)
            move = moves_list[idx]
        # после temperature_moves оставляем move, который вернул MCTS (argmax)

        trajectory.append((obs, pi_vec, player))
        board.push(move)
        position_deque.append(board_to_planes(board))
        moves_cnt += 1

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        z_white = 0.0
        winner = None
    else:
        if outcome.winner == chess.WHITE:
            z_white = 1.0
        else:
            z_white = -1.0

    data = []
    for obs, pi_vec, player in trajectory:
        z = z_white if player == chess.WHITE else -z_white
        data.append((obs, pi_vec, z))

    exceeded_max_moves = (moves_cnt == max_moves)
    return data, exceeded_max_moves


def train_one_iteration(model: CNNActorCritic,
                        optimizer: torch.optim.Optimizer,
                        num_simulations=64,
                        max_moves=200,
                        device="cpu"):
    """
    Одна итерация обучения:
      1) генерируем одну self-play партию (self_play_game),
      2) добавляем все позиции в replay buffer,
      3) делаем несколько SGD- шагов по случайным минибатчам из буфера.

    Возвращает словарь со статистикой.
    """

    # 1. self-play: одна партия
    data, exceeded_move_limit = self_play_game(model=model,
                                               num_simulations=num_simulations,
                                               max_moves=max_moves,
                                               device=device)
    if data is None or len(data) == 0:
        return None

    # 2. добавляем позиции партии в буфер
    replay_buffer.add_many(data)
    buffer_size = len(replay_buffer)

    # Если данных совсем мало — просто копим буфер, почти не обучаясь
    if buffer_size < BATCH_SIZE:
        return {
            "loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "num_positions": len(data),   # сколько позиций добавили
            "buffer_size": buffer_size,
            "train_steps": 0,
            "positions_used_for_training": 0,
            "exceeded move limit": None
        }

    # Сколько шагов SGD делаем на этой итерации
    if buffer_size < MIN_REPLAY_SIZE:
        # тёплый старт: буфер ещё небольшой → один шаг
        train_steps = 1
    else:
        train_steps = TRAIN_STEPS_PER_ITER

    last_loss = 0.0
    last_value_loss = 0.0
    last_policy_loss = 0.0
    total_positions_used = 0

    model.train()

    for _ in range(train_steps):
        obs_batch, pi_batch, z_batch = replay_buffer.sample(BATCH_SIZE)
        total_positions_used += len(z_batch)

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        pi_t = torch.as_tensor(pi_batch, dtype=torch.float32, device=device)
        z_t = torch.as_tensor(z_batch, dtype=torch.float32, device=device)

        logits, v_pred = model(obs_t)

        value_loss = F.mse_loss(v_pred, z_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().item())
        last_value_loss = float(value_loss.detach().item())
        last_policy_loss = float(policy_loss.detach().item())

    return {
        "loss": last_loss,
        "value_loss": last_value_loss,
        "policy_loss": last_policy_loss,
        "num_positions": len(data),                 # добавлено в буфер
        "buffer_size": buffer_size,                 # размер буфера после добавления
        "train_steps": train_steps,                 # сколько SGD-шагов
        "positions_used_for_training": total_positions_used,
        "exceeded_move_limit": exceeded_move_limit,
    }



class Node:
    def __init__(self, prior: float, player_to_move: bool):
        # Вероятность, что меня, а не другого ребёнка выберет родитель
        self.prior_probability = float(prior)
        # Сколько раз меня выбирали за всё время
        self.number_of_visits = 0
        # Оценка моего состояния
        self.value = 0.0
        # value / number_of_visits
        self.mean_value = 0.0
        self.children = {}
        # 1 = white, 0 = black
        self.player_to_move = player_to_move


class MCTS:
    def __init__(self,
                 number_of_simulations: int = 100,
                 c_puct: float = 1.5,
                 model: CNNActorCritic | None = None,
                 device: str = "cpu"):
        self.number_of_simulations = number_of_simulations
        # constant predictor upper confidence bound applied to trees
        # коэффициент исследования в формуле PUCT
        # Т.е. насколько сильно мы хотим исследовать
        # Чем больше константа, тем больше исследуем
        self.c_puct = c_puct

        self.model = model
        self.device = device
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _ucb_score(self, parent: Node, child: Node):
        # Если ещё никогда не были в вершине, то и оценки
        # у этой вершины нет
        if child.number_of_visits == 0:
            q = 0.0
        else:
            q = child.mean_value

        _eps = 1e-8
        u = (self.c_puct
             * child.prior_probability
             * math.sqrt(parent.number_of_visits + _eps)
             / (1.0 + child.number_of_visits))
        return q + u

    def run(self, root_state, add_dirichlet_noise: bool = False):
        """
        Главный метод:
        - root_state — chess.Board
        - Вернёт лучшую action и распределение policy по действиям.
        - Если add_dirichlet_noise=True, то в корне добавляем шум Дирихле
          к приорам (используется только в self-play).
        """
        root = Node(prior=1.0, player_to_move=self._player_to_move(root_state))

        dirichlet_applied = False

        for _ in range(self.number_of_simulations):
            self._simulate(root_state, root)

            # Как только у корня появились дети — один раз
            # добавляем к их prior'ам шум Дирихле
            if add_dirichlet_noise and (not dirichlet_applied) and root.children:
                self._add_dirichlet_noise(root)
                dirichlet_applied = True

        best_action = self._select_action_from_root(root)
        policy = self._policy_from_root(root)
        return best_action, policy

    def _add_dirichlet_noise(self, root: Node,
                             alpha: float = 0.3,
                             epsilon: float = 0.25):
        """
        Добавляем шум Дирихле к приорам в корне дерева.
        Используем только в self-play, чтобы дебюты были разнообразнее.

        root.children[move].prior_probability
        <- (1 - eps) * prior + eps * noise_i
        """
        if not root.children:
            return

        n = len(root.children)
        # генерируем вектор шума из Dirichlet(alpha)
        noise = np.random.dirichlet([alpha] * n)

        for child, n_i in zip(root.children.values(), noise):
            child.prior_probability = (
                (1.0 - epsilon) * child.prior_probability
                + epsilon * float(n_i)
            )


    def _player_to_move(self, state:chess.Board):
        """
        True = white, False = black (совпадает с chess.WHITE / chess.BLACK).
        """
        return state.turn

    def _simulate(self, state: chess.Board, root_node: Node):
        """
        Одна MCTS-симуляция.
        state: текущее состояние доски
        root_node: соответствующий узел в дереве
        - идти вниз по дереву (Selection),
        - возможно расширять (Expansion),
        - оценивать (Rollout/Evaluation),
        - поднимать value вверх (Backup).
        """
        board = state.copy()

        path = [root_node]
        node = root_node

        # Selection
        while True:
            if self._is_terminal(board):
                break
            if not node.children:
                break

            best_score = float("-inf")
            best_move = None
            best_child = None

            for move, child in node.children.items():
                score = self._ucb_score(node, child)
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_child = child

            assert best_child is not None, "MCTS: best_child is None, дерево в неконсистентном состоянии"

            board.push(best_move)
            node = best_child
            path.append(node)

        # Теперь board / node - лист (или терминальная позиция)

        # Expansion + evaluation

        if self._is_terminal(board):
            value = self._evaluate_terminal(board, root_node.player_to_move)
        else:
            legal_moves = self._get_legal_moves(board)
            assert legal_moves, "MCTS: нет легальных ходов в нетерминальной позиции"
            assert self.model is not None, "MCTS: модель не передали, получил None"

            obs = board_to_obs(board, position_deque)
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits, v_pred = self.model(obs_t)
                v = float(v_pred.item())
                probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)

            if board.turn != root_node.player_to_move:
                v = -v

            priors = {}
            total_p = 0.0

            for move in legal_moves:
                action_id = move_to_id(board, move)
                assert 0 <= action_id < TOTAL_MOVES, "move_to_id вернул некорректный id"
                p = float(probs[action_id])
                priors[move] = p
                total_p += p

            for move in legal_moves:
                child = Node(prior=priors[move],
                             player_to_move=(not node.player_to_move))
                node.children[move] = child

            value = v

        # backup: поднимаемся по пути вверх
        for node in path:
            node.number_of_visits += 1
            node.value += value
            node.mean_value = node.value / node.number_of_visits

    def _select_action_from_root(self, root: Node):
        """
        Выбираем ход с максимальным числом посещений.
        Пока считаем, что root.children уже заполнены.
        """
        if not root.children:
            return None # Если нет детей, то нет ходов

        best_move = None
        best_visits = -1

        for move, child in root.children.items():
            if child.number_of_visits > best_visits:
                best_visits = child.number_of_visits
                best_move = move

        return best_move

    def _policy_from_root(self, root: Node):
        """
        Строим распределение по ходам из корня:
        prob(move) пропорционально number of visits(child)
        """

        visits = {move: child.number_of_visits for move, child in root.children.items()}
        total = sum(visits.values())
        if total == 0:
            n = len(visits)
            return {move: 1.0 / n for move in visits.keys()}

        return {move: v / total for move, v in visits.items()}

    def _rollout(self, board: chess.Board, player_to_move_at_root: bool, max_depth: int = 100) -> float:
        """
        Простой rollout: играем случайными ходами до конца партии или max_depth.
        Возвращаем результат с точки зрения player_to_move_at_root.
        """
        b = board.copy()
        depth = 0
        while not b.is_game_over() and depth < max_depth:
            moves = self._get_legal_moves(b)
            if not moves:
                break
            move = random.choice(moves)
            b.push(move)
            depth += 1

        if not b.is_game_over():
            return 0.0

        outcome = b.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0

        winner = outcome.winner
        return 1.0 if winner == player_to_move_at_root else -1.0


    def _get_legal_moves(self, board: chess.Board):
        return list(board.legal_moves)

    def _is_terminal(self, board: chess.Board) -> bool:
        return board.is_game_over()

    def _evaluate_terminal(self, board: chess.Board, player_to_move: bool) -> float:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0

        winner = outcome.winner
        if winner == player_to_move:
            return 1.0
        else:
            return -1.0

    def _material(self, board: chess.Board, color: bool) -> int:
        total = 0
        for piece in board.piece_map().values():
            if piece.color == color:
                total += PIECE_VALUES[piece.piece_type]
        return total

    def _evaluate_position(self, board: chess.Board, root_player: bool) -> float:
        """
        Оценка нетерминальной позиции по материалу с точки зрения root_player
        Возвращает диапазон примерно [-1, 1].
        """
        assert not board.is_game_over()

        white_mat = self._material(board, chess.WHITE)
        black_mat = self._material(board, chess.BLACK)

        if root_player == chess.WHITE:
            diff = white_mat - black_mat
        else:
            diff = black_mat - white_mat

        # нормируем
        return max(-1.0, min(1.0, diff / 20.0))

    def _evaluate_position_nn(self, board: chess.Board, root_player: bool) -> float:
        """
        Оцениваем позицию нейросетью.
        Сеть обучена давать value с точки зрения игрока, который ходит.
        Здесь мы переводим это к точке зрения root_player.
        """
        assert self.model is not None

        obs = board_to_obs(board, position_deque)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            v_pred = self.model.values_only(obs_t)
            v = float(v_pred.item())

        if board.turn != root_player:
            v = -v
        return v
