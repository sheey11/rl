import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torch
import time

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

from config import *
import gameboard

from dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

board = gameboard.generate_board(BATCH_SIZE, DIFFICULTY, device)

net = DQN().to(device)
optimizer = optim.SGD(net.parameters(), lr=5e-5)
criterion = nn.MSELoss()

q_decay = 0.9

def train(net, board, optim, loss=nn.MSELoss(), episode=1e4, epsilon=0.7):
    losses = []

    finishes = []
    episodes = []

    EPISODE = int(episode)
    finished_counts = torch.tensor(0, device=device)

    net.train()
    pbar = tqdm(range(EPISODE))
    for e in pbar:
        q_values = net(board)
        if np.random.rand() < epsilon:
            actions = torch.randint(0, 4, (board.size(0),)).to(device)
        else:
            actions = torch.argmax(q_values, dim=1)

        reward, next_board = gameboard.move(board, actions)

        with torch.no_grad():
            next_q = net(next_board).detach()
        next_q = next_q.max(1)[0]
        next_q[reward == GAME_WIN_REWARD] = 0 # ignore next_q if already win

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        distance = gameboard.manhattan(next_board)
        desired_q = (reward - distance ** 2) + next_q * q_decay

        optimizer.zero_grad()
        loss = criterion(q_values, desired_q)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({
            'avg dist': f'{torch.mean(distance):.2f}',
            'finished': str(finished_counts.cpu().item()),
        })

        #board = next_board

        finished = distance == 0
        board[~finished] = next_board[~finished]
        if finished.any():
            board = board[~finished]
            # finished_counts += finished.sum()
            finished_counts = finished.sum()
        if (e + 1) % MAX_STEP == 0 or board.size(0) == 0:
            board = gameboard.generate_board(BATCH_SIZE, DIFFICULTY, device)
            finishes += [finished_counts.cpu().item()]
            episodes += [e]
            finished_counts = torch.tensor(0, device=device)

        losses.append(loss.detach().cpu().item())

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    sns.lineplot(x=range(EPISODE), y=losses, ax=ax1)
    sns.lineplot(x=episodes, y=finishes, ax=ax2)
    plt.show()

    torch.save(net.state_dict(), 'model.pt')

if __name__ == '__main__':
    train(net, board, optimizer)