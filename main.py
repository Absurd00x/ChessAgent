from agent import PPO
from torch import save, load
from env import ChessEnv
from nw import CNNActorCritic
import os

#windows kek
opponent = CNNActorCritic()

def make_env_function():
    return ChessEnv(model=opponent, desired_color=None, record=False)


os.makedirs("checkpoints", exist_ok=True)
agent = PPO(make_env_function)

my_path = "./checkpoints/second.pth"
opponent_path = "./checkpoints/first.pth"

if os.path.exists(my_path):
    data = load(my_path, map_location="cpu")
    agent.model.load_state_dict(data)
    print("My checkpoint loaded")
else:
    print("Current model file not found")

if os.path.exists(opponent_path):
    data = load(opponent_path, map_location="cpu")
    opponent.load_state_dict(data)
    print("Opponent loaded")
else:
    print("Opponent model file not found")

for p in opponent.parameters():
    p.requires_grad_(False)
opponent.eval()

while True:
    try:
        agent.train(desired_winrate=80, plot=True)
    except KeyboardInterrupt:
        print("Terminating...")
        break
    finally:
        save(agent.model.state_dict(), my_path)
