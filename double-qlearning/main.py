import numpy as np
import pandas as pd
import time

from agent import *

episode = 0
agent = Agent()

while True:
    episode += 1
    game_over = False
    state = (0, 0)
    step = 0
    reward = 0

    walked = set()

    while not game_over:
        step += 1
        print('step:', step)
        next_state, reward, action, game_over = agent.next_state(state)
        if next_state in walked:
            agent.update_q_table(state, action, -0.5, next_state)
        walked.add(next_state)
        state = next_state
        agent.draw(state)
        time.sleep(0.01)
    print('Episode: %d, total step: %d, %s' % (episode, step, 'Found' if reward > 0 else 'Dead'))
    time.sleep(0.5)