import random
from sys import setrecursionlimit

import torch
from torch import save, load
from nw import CNNActorCritic
import os
from agent import MCTS, board_to_obs, policy_to_pi_vector, self_play_game, train_one_iteration
import chess
from constants import CHECKPOINT_PATH
from replay_buffer import load_replay_buffer, save_replay_buffer

def play_game_mcts_vs_random(num_simulations=256, max_moves=200):
    board = chess.Board()
    mcts = MCTS(number_of_simulations=num_simulations)

    move_number = 1
    while not board.is_game_over() and move_number <= max_moves:
        print(f"\nMove {move_number}")
        print(board)

        if board.turn == chess.WHITE:
            move, policy = mcts.run(board)
            print("MCTS plays:", move)
        else:
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            print("Random plays:", move)

        board.push(move)
        move_number += 1

    print("\nFinal position:")
    print(board)
    print("Game over:", board.outcome())

def play_game_mcts_nn_vs_random(num_simulations=256, max_moves=200, device="cpu"):
    board = chess.Board()

    model = CNNActorCritic().to(device)
    state_dict = torch.load("checkpoints/alphazero_like.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    mcts = MCTS(number_of_simulations=num_simulations,
                model=model,
                device=device)

    move_number = 1
    while not board.is_game_over() and move_number <= max_moves:
        print(f"Move {move_number}")
        print(board)

        if board.turn == chess.WHITE:
            move, policy = mcts.run(board)
        else:
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            print("Random plays:", move)

        board.push(move)
        move_number += 1

    print("\nFinal position:")
    print(board)
    print("Game over:", board.outcome())


def debug_once():
    board = chess.Board()
    mcts = MCTS(number_of_simulations=64)
    move, policy = mcts.run(board)

    obs = board_to_obs(board)
    pi_vec = policy_to_pi_vector(board, policy)

    print("obs shape:", obs.shape)
    print("pi_vec shape:", pi_vec.shape)

    net = CNNActorCritic()
    x = torch.from_numpy(obs).unsqueeze(0)
    logits, value = net(x)
    print("logits shape:", logits.shape)
    print("value shape:", value.shape)


def debug_self_play():
    data = self_play_game(num_simulations=64, max_moves=200)
    print("Positons collected:", len(data))
    obs, pi_vec, z = data[0]
    print("obs shape:", obs.shape)
    print("pi_vec shape:", pi_vec.shape)
    print("z example", z)


def train_loop(num_iters=10, device="cpu"):
    model = CNNActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.to(device)
    for it in range(1, num_iters + 1):
        stats = train_one_iteration(model,
                                    optimizer,
                                    num_simulations=64,
                                    max_moves=200,
                                    device=device)
        print(f"Iter {it}:",
              f"loss={stats['loss']:.4f}",
              f"policy_loss={stats['policy_loss']:.4f}",
              f"positions={stats['num_positions']}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/alphazero_like.pth")

def main(device: str="cuda"):
    os.makedirs("checkpoints", exist_ok=True)

    model = CNNActorCritic().to(device)

    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print("Model not loaded. Creating new one...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Загрузка replay buffer, если файл существует
    print("Trying to load replay buffer...")
    if load_replay_buffer():
        print("Replay buffer loaded from disk.")
    else:
        print("No replay buffer found, starting with an empty buffer.")

    print("Starting infinite self-play training loop. Press Ctrl+C to stop.")

    i = 0
    try:
        while True:
            i += 1
            stats = train_one_iteration(
                model,
                optimizer,
                num_simulations=512,
                max_moves=200,
                device=device,
            )

            if stats is None:
                continue

            print(
                f"Iter {i}: "
                f"loss={stats['loss']:.8f} "
                f"policy_loss={stats['policy_loss']:.8f} "
                f"value_loss={stats['value_loss']:.8f} "
                f"positions={stats['num_positions']} "
                f"buffer={stats.get('buffer_size', 0)} "
                f"steps={stats.get('train_steps', 0)} "
                f"train_pos_used={stats.get('positions_used_for_training', 0)}"
            )

            # периодически сохраняем модель и replay buffer
            if i % 10 == 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"Checkpoint saved at iter {i}")
                save_replay_buffer()
                print(f"Replay buffer saved at iter {i}")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Saving final checkpoint and replay buffer...")
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        save_replay_buffer()
        print(f"Final model saved to {CHECKPOINT_PATH}")
        print("Replay buffer saved to replay_buffer.npz")

if __name__ == "__main__":
    # play_game_mcts_vs_random()
    # debug_once()
    # debug_self_play()
    # train_loop(num_iters=10, device="cuda")
    # play_game_mcts_nn_vs_random(device="cuda")
    main(device="cuda")

