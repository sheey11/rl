from typing import Tuple
import numpy as np
import pandas as pd

class Agent:
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    actions = np.array(['l', 'r', 'u', 'd'])

    env_map = [
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', 'x', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', 'x', '-', 'x', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', 'x', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'o', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-'],
        ['-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ]

    def __init__(self, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64, index=pd.MultiIndex.from_tuples([], names=['y', 'x']))

    def ensure_qtable(self, state: Tuple[int, int]):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.DataFrame(np.zeros((1, 4)), columns=self.actions, index=pd.MultiIndex.from_tuples([state], names=['y', 'x']))
            )

    # returns state, reward, gameover
    def next_state(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, str, bool]:
        self.ensure_qtable(state)
        y, x = state[0], state[1]

        act = ''

        if np.random.uniform() < self.epsilon:
            act = np.random.choice(self.actions)
        else:
            q_value = self.q_table.loc[state, :].values
            maxq = q_value.max()
            act = np.random.choice(self.actions[maxq == q_value])

        if act == 'l':
            x = x - 1 if x > 0 else x
        elif act == 'r':
            x = x + 1 if x < len(self.env_map[0]) - 1 else x
        elif act == 'u':
            y = y - 1 if y > 0 else y
        else:
            y = y + 1 if y < len(self.env_map) - 1 else y
        
        label = self.env_map[y][x]

        if label == 'o':
            reward = 1.0
        elif label == 'x':
            reward = -1.0
        else:
            reward = 0

        self.update_q_table(state, act, reward, (y, x))

        return (y, x), reward, act, reward != 0
    
    def update_q_table(self, state: Tuple[int, int], action, reward, next_state: Tuple[int, int]):
        self.ensure_qtable(next_state)
        q_predict = self.q_table.loc[state, action]
        max_q = self.q_table.loc[next_state, :].values.max()
        diff = self.alpha * (reward + self.gamma * max_q - q_predict)
        self.q_table.loc[state, action] += diff
    
    def draw(self, state):
        y, x = state
        env = [row.copy() for row in self.env_map]
        env[y][x] = 'A'
        lines = [ ''.join(env[i]) for i in range(len(env)) ]
        print('\n'.join(lines))
        print(''.join(['\033[F'] * (len(lines) + 2)))
