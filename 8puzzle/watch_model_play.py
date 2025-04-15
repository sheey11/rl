if __name__ != '__main__':
    exit(1)

import time
from gameboard import Gym, RewardConfig, Action
import torch
from models import FFNNetwork

grid_size = 3
device = torch.device('cuda', 0)
difficulty = 10

cfg = RewardConfig()
gym = Gym(grid_size, 1, cfg, difficulty=difficulty, device=device)
gym.init_boards()

model = FFNNetwork(grid_size)
model.load_state_dict(torch.load("ffn.pt"))
model = model.to(device)

print(gym)
while True:
    with torch.no_grad():
        predictions = model(gym.boards)
    actions = torch.argmax(predictions, dim=1)
    print(f"action: {['left', 'up', 'right', 'down'][actions[0]]}, {predictions}")
    reward = gym.move(actions)
    print(gym)

    if gym.win().all():
        print("Model wins!")
        exit()

    time.sleep(1)

