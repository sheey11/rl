from sarsa import *
import time

agent = Sarsa(expected_sarsa=True)

state = (0, 0)
episode = 1

while True:
    step = 0
    gameover = False
    action = agent.init_action(state)
    reward = 0
    state = (0, 0)
    while not gameover:
        state, action, reward, gameover = agent.next_step(state, action)
        step += 1
        print('step %d' % step)
        agent.show(state)
        time.sleep(0.01)
    print('episode %d, total step %d, state %s' % (episode, step, 'found' if reward > 0 else 'dead'))
    episode += 1