if __name__ != '__main__':
    exit(1)

from gameboard import Gym, RewardConfig, Action
import torch, getch

cfg = RewardConfig()
gym = Gym(3, 1, cfg)
gym.init_boards()

actions = {
    'h': torch.tensor([Action.LEFT]),
    'j': torch.tensor([Action.DOWN]),
    'k': torch.tensor([Action.UP]),
    'l': torch.tensor([Action.RIGHT]),
    'a': torch.tensor([Action.LEFT]),
    's': torch.tensor([Action.DOWN]),
    'w': torch.tensor([Action.UP]),
    'd': torch.tensor([Action.RIGHT]),
}

print(gym)
while True:
    print('[hjkl/wsad]: ', end="", flush=True)
    k = getch.getche()
    if k not in actions:
        continue
    reward = gym.move(actions[k])

    print('\n')
    print(f"You rewarded {reward[0]}.")
    print(gym)
