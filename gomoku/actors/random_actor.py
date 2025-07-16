from typing import Any
from gomoku import GomokuEnv
import numpy as np
import random


class RandomActor:
    def __call__(self, env: GomokuEnv) -> Any:
        empty_positions = np.argwhere(env.board == 0)
        if len(empty_positions) == 0:
            raise ValueError("No valid moves left.")

        # 随机选择一个空位置
        row, col = random.choice(empty_positions)
        return int(row), int(col)
