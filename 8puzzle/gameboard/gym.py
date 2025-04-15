import numpy as np
import numpy.typing as npt
import torch
from dataclasses import dataclass
import enum


class Action(enum.IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


@dataclass
class RewardConfig:
    win: int = 10
    invalid_action: int = -1


class Gym:
    def __init__(
        self,
        grid_size: int,
        batch_size: int,
        reward_config: RewardConfig,
        difficulty: int = 5,
        device=None,
    ):
        self.grid_size = grid_size
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.difficulty = difficulty
        self.boards = self.get_final(reshape_to_grid=True)
        self.reward_config = reward_config

    def move(self, actions: torch.Tensor):
        # left, up, right, down
        action_coords_shift_x = torch.tensor([1, 0, -1, 0], device=self.device)
        action_coords_shift_y = torch.tensor([0, 1, 0, -1], device=self.device)

        desire = self.get_final()

        flatten = self.boards.view(-1, self.grid_size**2)
        zero_coord_ys, zero_coord_xs = (
            torch.div(flatten.argmin(dim=1), self.grid_size, rounding_mode="floor"),
            flatten.argmin(dim=1) % self.grid_size,
        )

        ys, xs = (
            zero_coord_ys + action_coords_shift_y[actions],
            zero_coord_xs + action_coords_shift_x[actions],
        )
        invalid_actions = (
            (ys < 0) | (ys >= self.grid_size) | (xs < 0) | (xs >= self.grid_size)
        )
        ys[invalid_actions], xs[invalid_actions] = (
            zero_coord_ys[invalid_actions],
            zero_coord_xs[invalid_actions],
        )

        value_mask = (~invalid_actions).type(torch.int)
        zero_mask = invalid_actions.type(torch.int)
        value = self.boards[torch.arange(actions.size(0), device=self.device), ys, xs]
        self.boards[torch.arange(actions.size(0), device=self.device), ys, xs] = value * zero_mask
        self.boards[torch.arange(actions.size(0), device=self.device), zero_coord_ys, zero_coord_xs] = value * value_mask

        reward = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)
        reward[invalid_actions] = self.reward_config.invalid_action
        flatten = self.boards.view(-1, self.grid_size**2)
        win_mask = (flatten == desire).all(dim=1)
        reward[win_mask] = self.reward_config.win

        return reward

    def init_boards(self):
        self.boards = self.generate_board(self.grid_size, self.batch_size, self.difficulty, self.device)

    def hamming(self):
        desire = self.get_final()
        flatten = self.boards.view(-1, self.grid_size**2)
        return (flatten != desire).sum(dim=1).float()

    def manhattan(self):
        boards = self.boards.clone()
        boards[boards == 0] = 9
        boards -= 1
        row = boards // self.grid_size
        col = boards % self.grid_size
        std_row = torch.arange(0, self.grid_size).view(-1, 1).repeat([1, self.grid_size]).to(self.device)
        std_col = std_row.T

        dx = torch.abs(row - std_row)
        dy = torch.abs(col - std_col)
        d = (dx + dy).view(dx.size(0), -1)
        return torch.sum(d, dim=1)

    @staticmethod
    def generate_board(grid_size: int, batch_size: int, difficulty: int, device):
        gym = Gym(
            grid_size,
            batch_size,
            RewardConfig(),
            difficulty,
            device,
        )

        d = torch.zeros(batch_size, device=device)
        ready_boards = []
        while len(ready_boards) < batch_size:
            actions = torch.randint(0, 4, (gym.batch_size,)).to(device)
            _ = gym.move(actions)
            d = gym.manhattan()
            ready_mask = d >= difficulty

            ready_board = gym.boards[ready_mask]
            if len(ready_board) > 0:
                ready_boards.extend(ready_board)

            gym.boards = gym.boards[~ready_mask]
            if len(ready_board) > 0:
                gym.boards = torch.cat([gym.boards, gym.get_final(len(ready_board), True)])

        return torch.stack(ready_boards[:batch_size])

    def __str__(self):
        if self.boards.size(0) != 1:
            return f"<PuzzleGym(grid_size={self.grid_size}, batch_size={self.boards.size(0)})>"

        result = ""
        head = "┏" + "━━━┳" * (self.grid_size - 1) + "━━━┓"
        middle = "┣" + "━━━╋" * (self.grid_size - 1) + "━━━┫"
        tail = "┗" + "━━━┻" * (self.grid_size - 1) + "━━━┛"

        result += head + "\n"

        for i in range(self.grid_size):
            result += "┃"
            for j in range(self.grid_size):
                number = str(self.boards[0, i, j].item())
                number = " " if number == "0" else number
                result += "{:^3}┃".format(number)
            result += f"\n{middle}" if i != self.grid_size - 1 else ""
            result += "\n"

        result += tail
        return result
    
    def __repl__(self):
        return self.__str__()

    def get_final(self, n=None, reshape_to_grid=False):
        """return the boards of winning"""
        if n is None:
            n = self.batch_size
        final = np.roll(np.arange(self.grid_size**2), -1)
        if reshape_to_grid:
            final = final.reshape(1, self.grid_size, self.grid_size)
        else:
            final = final.reshape(1, -1)
        final = final.repeat(n, axis=0)
        final = torch.tensor(final, device=self.device, dtype=torch.int64)
        return final
    
    def win(self):
        flatten = self.boards.flatten(1)
        desire = self.get_final()
        return (flatten == desire).all(dim=1).reshape((-1))

    def remove_win_and_fill_with_new(self):
        flatten = self.boards.flatten(1)
        desire = self.get_final()
        mask = (flatten == desire).all(dim=1).reshape((-1))

        n_win = int(mask.sum().item())
        if n_win == 0:
            return 0

        self.boards = self.boards[~mask]
        new_boards = self.generate_board(self.grid_size, n_win, self.difficulty, self.device)
        self.boards = torch.concat([self.boards, new_boards])

        return n_win
