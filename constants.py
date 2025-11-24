import chess

BOARD_LAYERS = 12
EN_PASSANT_LAYERS = 1
CASTLES_LAYERS = 4
TURN_LAYERS = 1
LAST_POSITIONS = 8
META_LAYERS = EN_PASSANT_LAYERS + CASTLES_LAYERS + TURN_LAYERS
TOTAL_LAYERS = META_LAYERS + LAST_POSITIONS * BOARD_LAYERS
CHECKPOINT_PATH = "checkpoints/alphazero_like.pth"

TOTAL_MOVES = 4672

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,  # короля обычно не считаем в материале
}

# ================== Гиперпараметры буфера ==================

# Примерно по памяти:
#   одна позиция:
#     obs: TOTAL_LAYERS*8*8 float32  ≈ 102*64*4B ≈ 26 КБ
#     pi:  TOTAL_MOVES float32       ≈ 4672*4B   ≈ 18 КБ
#   итог ~44 КБ на позицию.
# При CAPACITY=20_000 это ~880 МБ ОЗУ.
REPLAY_CAPACITY = 100_000       # максимум позиций в буфере
MIN_REPLAY_SIZE = 5_000        # с какого размера буфера начинаем full-обучение
BATCH_SIZE = 512               # размер минибатча
TRAIN_STEPS_PER_ITER = 32      # сколько SGD-шагов на одну итерацию

# ===========================================================
