from random import choice

import gymnasium as gym
import numpy as np
import chess
from collections import deque
from random_agent import RAgent

BOARD_LAYERS = 12
EN_PASSANT_LAYERS = 1
CASTLES_LAYERS = 4
TURN_LAYERS = 1
LAST_POSITIONS = 8
META_LAYERS = EN_PASSANT_LAYERS + CASTLES_LAYERS + TURN_LAYERS
TOTAL_LAYERS = META_LAYERS + LAST_POSITIONS * BOARD_LAYERS

TOTAL_MOVES = 4672


class ChessEnv(gym.Env):

    def __init__(self, model=None, desired_color=None, record=False):
        super().__init__()

        self.position_deque = deque(maxlen=LAST_POSITIONS)
        self.record = record
        self.recorded_positions = []

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(TOTAL_LAYERS, 8, 8), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(TOTAL_MOVES)
        self.board = chess.Board()

        if model is None:
            self.opponent = RAgent(self.board)
        else:
            self.opponent = model

        if desired_color is None:
            desired_color = choice((chess.WHITE, chess.BLACK))
        self.trainer_color = chess.BLACK if desired_color == chess.WHITE else chess.WHITE

    def _mask_legal(self):
        mask = np.zeros(TOTAL_MOVES, dtype=np.bool)
        for move in self.board.legal_moves:
            a = self._encode_action(move)
            b = self._decode_action(a)
            assert b == move
            mask[a] = np.bool(True)
        return mask

    def reset(self, seed=None, options=None):
        super().reset()
        self.board.reset()
        info = {}
        if self.record:
            self.recorded_positions.clear()
        self.position_deque.clear()
        self.position_deque.append(self._encode_pieces())

        obs = self._encode_obs()

        if self.trainer_color == chess.WHITE:
            action = self.opponent(obs)
            if isinstance(action, int):
                action = self._decode_action(action)
            self.board.push(action)
            obs = self._encode_obs()

        info["legal_mask"] = self._mask_legal()
        return obs, info

    def step(self, action):
        # Ход нейросети
        move = self._decode_action(action)

        # нелегальный ход
        if move not in self.board.legal_moves:
            obs = self._encode_obs()
            return obs, -0.1, False, False, {"illegal": True}

        # применяем ход
        if self.board.is_capture(move):
            reward = 0.01
        else:
            reward = 0
        self.board.push(move)
        # print(self.board)

        if self.record:
            self.recorded_positions.append(move.uci())

        # добавляем новую 12-слойную карту фигур в deque
        new_base = self._encode_pieces()
        self.position_deque.append(new_base)

        if self.board.is_game_over():
            obs = self._encode_obs()
            outcome = self.board.outcome()
            if outcome.winner is None:
                reward = 0.0
            else:
                reward = 1.0
            return obs, reward, True, False, {"winner": True}

        obs = self._encode_obs()

        # Ход оппонента
        action = self.opponent(obs)
        if isinstance(action, int):
            action = self._decode_action(action)
        self.board.push(action)

        if self.board.is_game_over():
            obs = self._encode_obs()
            outcome = self.board.outcome()
            if outcome.winner is None:
                reward = 0.0
            else:
                reward = -1.0
            return obs, reward, True, False, {"winner": False}

        info = {}
        mask = self._mask_legal()
        info["legal_mask"] = mask
        return obs, reward, False, False, info

    def _encode_pieces(self) -> np.ndarray:
        planes = np.zeros((BOARD_LAYERS, 8, 8), dtype=np.float32)
        piece_map = self.board.piece_map()
        mapping = {
            (chess.PAWN,   True): 0,
            (chess.KNIGHT, True): 1,
            (chess.BISHOP, True): 2,
            (chess.ROOK,   True): 3,
            (chess.QUEEN,  True): 4,
            (chess.KING,   True): 5,
            (chess.PAWN,   False): 6,
            (chess.KNIGHT, False): 7,
            (chess.BISHOP, False): 8,
            (chess.ROOK,   False): 9,
            (chess.QUEEN,  False): 10,
            (chess.KING,   False): 11,
        }

        for square, piece in piece_map.items():
            p = mapping[(piece.piece_type, piece.color)]
            r = chess.square_rank(square)
            c = chess.square_file(square)
            planes[p, r, c] = 1.0
        return planes

    def _encode_obs(self):
        obs = np.zeros((TOTAL_LAYERS, 8, 8), dtype=np.float32)

        idx = 0

        if self.board.turn == chess.WHITE:
            obs[idx, :, :] = 1.0
        idx += 1

        # рокировки: WK, WQ, BK, BQ
        obs[idx, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.WHITE) else 0.0
        idx += 1
        obs[idx, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.WHITE) else 0.0
        idx += 1
        obs[idx, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.BLACK) else 0.0
        idx += 1
        obs[idx, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.BLACK) else 0.0
        idx += 1

        # эн-пассант
        if self.board.ep_square is not None:
            r = chess.square_rank(self.board.ep_square)
            c = chess.square_file(self.board.ep_square)
            obs[idx, r, c] = 1.0
        idx += 1

        for b in self.position_deque:
            obs[idx: idx + BOARD_LAYERS] = b
            idx += BOARD_LAYERS

        return obs

    def _decode_action(self, action_id: int) -> chess.Move:
        """
        Декодируем индекс из [0, 4671] в ход python-chess (chess.Move).

        Схема:
        - 64 from-клетки * 73 плоскости.
        - 0..55 : 8 направлений * 7 расстояний (слайдинги).
        - 56..63: 8 ходов коня.
        - 64..72: 9 underpromotion (N/B/R) для пешек.
        """

        if action_id < 0 or action_id >= TOTAL_MOVES:
            return chess.Move.null()

        from_sq = action_id // 73              # 0..63
        plane = action_id % 73                # 0..72

        from_file = chess.square_file(from_sq)  # 0..7
        from_rank = chess.square_rank(from_sq)  # 0..7

        # ---------- 0..55: слайдинги ----------
        if plane < 56:
            dir_id = plane // 7               # 0..7
            k = plane % 7 + 1                 # 1..7

            # (df, dr): file, rank
            directions = [
                (1, 0),   # вправо
                (-1, 0),  # влево
                (0, 1),   # вверх
                (0, -1),  # вниз
                (1, 1),   # вверх-вправо
                (-1, 1),  # вверх-влево
                (1, -1),  # вниз-вправо
                (-1, -1), # вниз-влево
            ]
            df, dr = directions[dir_id]
            to_file = from_file + df * k
            to_rank = from_rank + dr * k

            # вышли за доску — считаем как null, потом отфильтруешь по legal_moves
            if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
                return chess.Move.null()

            to_sq = chess.square(to_file, to_rank)

            # особый случай: если это пешка, которая дошла до конца — авто-промо в ферзя
            piece = self.board.piece_at(from_sq)
            promotion = None
            if piece is not None and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE and to_rank == 7:
                    promotion = chess.QUEEN
                elif piece.color == chess.BLACK and to_rank == 0:
                    promotion = chess.QUEEN

            return chess.Move(from_sq, to_sq, promotion=promotion)

        # ---------- 56..63: конь ----------
        if plane < 64:
            knight_id = plane - 56
            knight_moves = [
                (1, 2), (2, 1),
                (2, -1), (1, -2),
                (-1, -2), (-2, -1),
                (-2, 1), (-1, 2),
            ]
            df, dr = knight_moves[knight_id]
            to_file = from_file + df
            to_rank = from_rank + dr

            if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
                return chess.Move.null()

            to_sq = chess.square(to_file, to_rank)
            return chess.Move(from_sq, to_sq)

        # ---------- 64..72: underpromotion N/B/R ----------
        promo_id = plane - 64                 # 0..8
        dir_id = promo_id // 3                # 0..2
        piece_id = promo_id % 3               # 0..2

        # под что промоутим (без ферзя: он покрыт "обычными" ходами выше)
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promo_piece = promo_pieces[piece_id]

        # три направления для пешки
        # для белых: вперёд, взятие влево, взятие вправо
        # для чёрных: зеркально
        if self.board.turn == chess.WHITE:
            directions = [
                (0, 1),   # вперёд
                (-1, 1),  # взятие влево
                (1, 1),   # взятие вправо
            ]
            final_rank = 7
        else:
            directions = [
                (0, -1),
                (1, -1),
                (-1, -1),
            ]
            final_rank = 0

        df, dr = directions[dir_id]
        to_file = from_file + df
        to_rank = from_rank + dr

        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return chess.Move.null()

        # underpromotion имеет смысл только с предпоследней горизонтали
        piece = self.board.piece_at(from_sq)
        if piece is None or piece.piece_type != chess.PAWN:
            return chess.Move.null()

        if self.board.turn == chess.WHITE and from_rank != 6:
            return chess.Move.null()
        if self.board.turn == chess.BLACK and from_rank != 1:
            return chess.Move.null()

        if to_rank != final_rank:
            return chess.Move.null()

        to_sq = chess.square(to_file, to_rank)
        return chess.Move(from_sq, to_sq, promotion=promo_piece)

    def _encode_action(self, move: chess.Move) -> int:
        """
        Обратное преобразование chess.Move -> action_id в [0, TOTAL_MOVES).

        Схема такая же, как в _decode_action:
        - 0..55  : слайдинги (8 направлений * 7 расстояний)
        - 56..63 : конь
        - 64..72 : underpromotion (N/B/R) пешек
        """

        # null-ход кодировать не будем
        if move is None or move == chess.Move.null():
            return -1

        from_sq = move.from_square
        to_sq = move.to_square

        if from_sq is None or to_sq is None:
            return -1

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)

        df = to_file - from_file
        dr = to_rank - from_rank

        # ---------- 1) Underpromotion N/B/R (плоскости 64..72) ----------
        if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

            # направления должны совпадать с _decode_action
            if self.board.turn == chess.WHITE:
                directions = [
                    (0, 1),  # вперёд
                    (-1, 1),  # взятие влево
                    (1, 1),  # взятие вправо
                ]
                final_rank = 7
                start_rank = 6
            else:
                directions = [
                    (0, -1),
                    (1, -1),
                    (-1, -1),
                ]
                final_rank = 0
                start_rank = 1

            # базовая валидация, как в _decode_action
            if chess.square_rank(from_sq) != start_rank or chess.square_rank(to_sq) != final_rank:
                return -1

            step = (df, dr)
            if step not in directions:
                return -1

            dir_id = directions.index(step)
            piece_id = promo_pieces.index(move.promotion)

            promo_id = dir_id * 3 + piece_id  # 0..8
            plane = 64 + promo_id  # 64..72

            action_id = from_sq * 73 + plane
            return action_id if 0 <= action_id < TOTAL_MOVES else -1

        # ---------- 2) Конь (плоскости 56..63) ----------
        knight_moves = [
            (1, 2), (2, 1),
            (2, -1), (1, -2),
            (-1, -2), (-2, -1),
            (-2, 1), (-1, 2),
        ]

        step = (df, dr)
        if step in knight_moves:
            knight_id = knight_moves.index(step)  # 0..7
            plane = 56 + knight_id  # 56..63
            action_id = from_sq * 73 + plane
            return action_id if 0 <= action_id < TOTAL_MOVES else -1

        # ---------- 3) Слайдинги (плоскости 0..55) ----------
        # те же directions, что в _decode_action
        directions = [
            (1, 0),  # вправо
            (-1, 0),  # влево
            (0, 1),  # вверх
            (0, -1),  # вниз
            (1, 1),  # вверх-вправо
            (-1, 1),  # вверх-влево
            (1, -1),  # вниз-вправо
            (-1, -1),  # вниз-влево
        ]

        # проверяем, что ход действительно "линейный"
        if df == 0 and dr == 0:
            return -1

        # шаг по направлению
        def sign(x: int) -> int:
            return (x > 0) - (x < 0)

        step = (sign(df), sign(dr))

        if step not in directions:
            # не слайдинг (и не конь, и не underpromotion) — не кодируем
            return -1

        # расстояние
        k_file = abs(df) if df != 0 else 0
        k_rank = abs(dr) if dr != 0 else 0
        k = max(k_file, k_rank)

        if k < 1 or k > 7:
            return -1

        dir_id = directions.index(step)  # 0..7
        plane = dir_id * 7 + (k - 1)  # 0..55

        action_id = from_sq * 73 + plane
        return action_id if 0 <= action_id < TOTAL_MOVES else -1

def make_env_function(model=None, desired_color=None, record=False):
    return ChessEnv(model=model, desired_color=desired_color, record=record)