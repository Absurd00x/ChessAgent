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
    chess.KING: 0,  # короля обычно не считаем в материале
}

TRAINING_MCTS_SIMULATIONS=800
TRAINING_MAX_MOVES=100_000
INFERENCE_BATCH_SIZE = 256
TEMPERATURE_MOVES = 32

# ================== Гиперпараметры буфера ==================

REPLAY_CAPACITY = 700_000       # максимум позиций в буфере
MIN_REPLAY_SIZE = 30_000        # с какого размера буфера начинаем full-обучение
BATCH_SIZE = 256               # размер минибатча
TRAIN_STEPS_PER_ITER = 16      # сколько SGD-шагов на одну итерацию
DEFAULT_REPLAY_PATH = "replay_buffer/replay_buffer.npz"
# ===========================================================

WEB_MCTS_SIMULATIONS = 800