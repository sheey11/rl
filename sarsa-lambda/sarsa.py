from functools import reduce
from typing import Tuple, List
import numpy as np
import pandas as pd

class Sarsa:

    actions = ['left', 'right', 'up', 'down']
    # maze = [
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', 'x', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'o', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-'],
    #     ['-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    #     ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    # ]

    maze = [
        ['-', '-', '-', '-'],
        ['-', '-', 'x', '-'],
        ['-', 'x', 'o', '-'],
        ['-', '-', '-', '-'],
    ]

    expected_sarsa = False

    path: List[Tuple[Tuple[int, int], str]] = []

    def __init__(self, lr:float=0.01, epsilon:float=0.1, gamma:float=0.01, expected_sarsa=False, lambda_:float=0.8):
        self.q_table = pd.DataFrame(columns = self.actions, index = pd.MultiIndex.from_tuples([], names=['y', 'x']), dtype = np.float64)
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.expected_sarsa = expected_sarsa
        self.lambda_ = lambda_

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
            reward = 100
            gameover = True
        elif self.maze[y][x] == 'x':
            reward = -100
            gameover = True
        else:
            reward = -1
            gameover = False

        return (y, x), reward, gameover

    def init_action(self, init_state=(0, 0)) -> str:
        action = self.choose_action(init_state)
        self.path = [(init_state, action)]
        return action

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

    def sarsa_diff(self, state, action, reward, next_state, next_action) -> float:
        return self.lr * (reward + self.gamma * self.q_table.loc[next_state, next_action] - self.q_table.loc[state, action])

    def exp_sarsa_diff(self, state, action, reward, next_state, next_action):
        action_propability = [0.25, 0.25, 0.25, 0.25] # randomly choose one action, so the probability is 1/4
        expectation: float = 0
        for i in range(len(self.actions)):
            expectation += action_propability[i] * self.q_table.loc[next_state, self.actions[i]]
        return self.lr * (reward + self.gamma * expectation - self.q_table.loc[state, action])

    def learn_one_step(self, state, action, reward, next_state, next_action) -> None:
        if self.expected_sarsa:
            diff = self.exp_sarsa_diff(state, action, reward, next_state, next_action)
        else:
            diff = self.sarsa_diff(state, action, reward, next_state, next_action)
        self.q_table.loc[state, action] += diff

    def learn(self, reward: float):
        if len(self.path) <= 1:
            return
        for i in range(len(self.path) - 2, -1, -1):
            state, action = self.path[i]
            next_state, next_action = self.path[i + 1]
            self.learn_one_step(state, action, reward, next_state, next_action)
            reward *= self.lambda_
        self.path = []

    def next_step(self, state, action) -> Tuple[Tuple[int, int], str, float, bool]:
        next_state, reward, gameover = self.take_action(state, action)
        next_action = self.choose_action(next_state)

        self.ensure_q_table(state)
        self.ensure_q_table(next_state)

        self.path.append((state, action))
        if gameover:
            self.path.append((next_state, next_action))

        self.learn(reward)

        return next_state, next_action, reward, gameover

    def show(self, state):
        map_copy = [row[:] for row in self.maze]
        y, x = state
        map_copy[y][x] = 'A'
        print('\n'.join([''.join(row) for row in map_copy]))
        print('\033[F' * (len(self.maze) + 2))