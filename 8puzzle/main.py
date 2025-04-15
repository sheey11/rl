import torch
import numpy as np
from tqdm import tqdm
from gameboard import Gym, RewardConfig
from collections import namedtuple
import random

from models import FFNNetwork

epochs = 10000
batch_size = 50
sample_batch_size = 100
grid_size = 3
explor_rate = 0.8
max_step = 20
memory_capacity = 1000
epsilon = 0.5
record_interval = 200
difficulty = 10
decay = 0.9
lr = 1e-4

device = torch.device("cuda", 0)
model = FFNNetwork(grid_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

gym = Gym(grid_size, batch_size, RewardConfig(), difficulty=difficulty, device=device)

Transition = namedtuple('Transition', ('state', 'action', 'new_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        #"""Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(memory_capacity)
def train():
    for epoch in tqdm(range(epochs)):
        gym.init_boards()
        for step in range(max_step):
            with torch.no_grad():
                state = gym.boards.clone()
                q_values = model(state)
                if np.random.random() < epsilon:
                    actions = torch.randint(0, 4, (batch_size,)).to(device)
                else:
                    actions = torch.argmax(q_values, dim=1)
                
                # take action and get reward
                rewards = gym.move(actions) - gym.manhattan()

                new_state = gym.boards.clone()
                for board, action, new_board, reward in zip(state, actions, new_state, rewards):
                    memory.push(board, action, new_board, reward)
                
            if len(memory) < sample_batch_size:
                continue

            transitions = memory.sample(sample_batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.stack(batch.state)
            action_batch = torch.stack(batch.action)
            new_state_batch = torch.stack(batch.new_state)
            reward_batch = torch.stack(batch.reward)

            non_final_state = (reward_batch != 10)
            predicted_rewards = model(state_batch).gather(1, action_batch.view(-1, 1)).view(-1)

            new_q = model(new_state_batch)
            max_q = torch.max(new_q, dim=1).values

            y = reward_batch.to(torch.float)
            y[non_final_state] += decay * max_q[non_final_state]

            loss = criterion(predicted_rewards, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train()
for g in optimizer.param_groups:
    g['lr'] = 1e-5
train()
for g in optimizer.param_groups:
    g['lr'] = 5e-5
train()

torch.save(model.state_dict(), "ffn.pt")
