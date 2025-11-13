from agent import PPO
from torch import save, load
import os

os.makedirs("checkpoints", exist_ok=True)

agent = PPO()
path = "./checkpoints/first.pth"

if os.path.exists(path):
    data = load(path, map_location="cpu")
    agent.model.load_state_dict(data)
    print("Checkpoint loaded")
else:
    print("Model file not found")

while True:
    try:
        agent.train(desired_winrate=80, plot=True)
    except KeyboardInterrupt:
        print("Terminating...")
        break
    finally:
        save(agent.model.state_dict(), path)
