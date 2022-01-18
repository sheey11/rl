from typing import Tuple
import numpy as np
import pandas as pd

class Sarsa:

    actions = ['left', 'right', 'up', 'down']
    maze = [
        ['-', '-', '-', '-'],
        ['-', '-', 'x', '-'],
        ['-', 'x', 'o', '-'],
        ['-', '-', '-', '-'],
    ]

    def __init__(self, lr:float=0.01, epsilon:float=0.1, gamma:float=0.01):
        self.q_table = pd.DataFrame(columns = self.actions, index = pd.MultiIndex.from_tuples([], names=['y', 'x']), dtype = np.float64)
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma

    def ensure_q_table(self, state: Tuple[int, int]):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.DataFrame(np.zeros((1, 4)), index=pd.MultiIndex.from_tuples([state]), columns=self.actions)
            )

    def take_action(self, state, action: str) -> Tuple[Tuple[int, int], float, bool]:
        y, x = state
        if action == 'left':
            x = max(0, x - 1)
        elif action == 'right':
            x = min(x + 1, len(self.maze[0]) - 1)
        elif action == 'up':
            y = max(0, y - 1)
        elif action == 'down':
            y = min(y + 1, len(self.maze) - 1)

        if self.maze[y][x] == 'o':
            reward = 1
            gameover = True
        elif self.maze[y][x] == 'x':
            reward = -1
            gameover = True
        else:
            reward = 0
            gameover = False

        return (y, x), reward, gameover

    def choose_action(self, state) -> str:
        self.ensure_q_table(state)
        if np.random.rand() < self.epsilon:
            next_action = np.random.choice(self.actions)
        else:
            q_values = self.q_table.loc[state, :].values
            max_q = q_values.max()
            actions = np.array(self.actions)[q_values == max_q]
            next_action = np.random.choice(actions)
        return next_action

    def learn(self, state, action, reward, next_state, next_action) -> None:
        diff = self.lr * (reward + self.gamma * self.q_table.loc[next_state, next_action] - self.q_table.loc[state, action])
        self.q_table.loc[state, action] += diff

    def next_step(self, state, action) -> Tuple[Tuple[int, int], str, float, bool]:
        next_state, reward, gameover = self.take_action(state, action)
        next_action = self.choose_action(next_state)

        self.ensure_q_table(state)
        self.ensure_q_table(next_state)
        self.learn(state, action, reward, next_state, next_action)

        return next_state, next_action, reward, gameover

    def show(self, state):
        map_copy = [row[:] for row in self.maze]
        y, x = state
        map_copy[y][x] = 'A'
        print('\n'.join([''.join(row) for row in map_copy]))
        print('\033[F' * (len(self.maze) + 2))