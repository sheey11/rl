import numpy as np
import torch

from config import *

# return the boards of winning
def get_final(batch_size, device, reshape=False):
    final = np.roll(np.arange(GRID_SIZE ** 2), -1)
    if reshape:
        final = final.reshape(1, GRID_SIZE, GRID_SIZE)
    else:
        final = final.reshape(1, -1)
    final = final.repeat(batch_size, axis=0)
    final = torch.tensor(final, device=device, dtype=torch.int64)
    return final

# 0: left, 1: up, 2: right, 3: down

def move(boards, actions):
    action_coords_shift_x = torch.tensor([1, 0, -1, 0], device=boards.device)
    action_coords_shift_y = torch.tensor([0, 1, 0, -1], device=boards.device)

    desire = get_final(boards.shape[0], boards.device)

    boards = boards.clone()
    flatten = boards.view(-1, GRID_SIZE ** 2)
    zero_coord_ys, zero_coord_xs = torch.div(flatten.argmin(dim=1), GRID_SIZE, rounding_mode='floor'), flatten.argmin(dim=1) % GRID_SIZE

    ys, xs = zero_coord_ys + action_coords_shift_y[actions], zero_coord_xs + action_coords_shift_x[actions]
    invalid_actions = (ys < 0) | (ys >= GRID_SIZE) | (xs < 0) | (xs >= GRID_SIZE)
    ys[invalid_actions], xs[invalid_actions] = zero_coord_ys[invalid_actions], zero_coord_xs[invalid_actions]

    value_mask = (~invalid_actions).type(torch.int)
    zero_mask = invalid_actions.type(torch.int)
    value = boards[range(boards.size(0)), ys, xs]
    boards[range(boards.size(0)), ys, xs] = value * zero_mask
    boards[range(boards.size(0)), zero_coord_ys, zero_coord_xs] = value * value_mask

    reward = torch.zeros(boards.size(0), dtype=torch.int, device=boards.device)
    reward[invalid_actions] = -100
    flatten = boards.view(-1, GRID_SIZE ** 2)
    win_mask = (flatten == desire).all(dim=1)
    reward[win_mask] = GAME_WIN_REWARD
    return reward, boards

def hamming(boards):
    desire = get_final(boards.shape[0], boards.device)
    flatten = boards.view(-1, GRID_SIZE ** 2)
    return (flatten != desire).sum(dim=1).float()

def manhattan(boards):
    flatten = boards.view(-1, GRID_SIZE ** 2)
    zero_coords = flatten.argmin(dim=1)

    dist_table = torch.arange(1, GRID_SIZE ** 2 + 1).to(boards.device)
    flat_dist = torch.abs(flatten - dist_table)
    mdist_y = torch.div(flat_dist, GRID_SIZE, rounding_mode='floor')
    mdist_x = flat_dist % GRID_SIZE
    mdist = mdist_x + mdist_y
    mdist[range(boards.size(0)), zero_coords] = 0
    return torch.sum(mdist, dim=1).view(-1).float()

def generate_board(n, distance, device):
    boards = get_final(n, device, reshape=True)
    d = torch.zeros(n, device=device)
    while (d < distance).any():
        mask = d < distance
        action = torch.randint(0, 4, (n,)).to(device)
        _, boards[mask] = move(boards[mask], action[mask])
        d = manhattan(boards)
    return boards

def display(board):
    head   = '┏' + '━━━┳' * (GRID_SIZE - 1) + '━━━┓'
    middle = '┣' + '━━━╋' * (GRID_SIZE - 1) + '━━━┫'
    tail   = '┗' + '━━━┻' * (GRID_SIZE - 1) + '━━━┛'
    print(head)
    for i in range(GRID_SIZE):
        print('┃', end='')
        for j in range(GRID_SIZE):
            number = str(board[i, j].item())
            number = ' ' if number == '0' else number
            print('{:^3}┃'.format(number), end='')
        print(f'\n{middle}' if i != GRID_SIZE -1 else '')
    print(tail)