# gomoku_env.py
import numpy as np
import gym
from gym import spaces
import pygame
from pygame.locals import *

class GomokuEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=15):
        self.board_size = board_size
        self.action_space = spaces.Box(
            low=0, high=board_size - 1, shape=(2,), dtype=int
        )  # 动作空间为 (row, col)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.uint8
        )
        self.board = np.zeros(
            (board_size, board_size), dtype=np.uint8
        )  # 初始化为空棋盘
        self.current_player = 1  # 1 for player 1 (black), 2 for player 2 (white)
        self.game_over = False  # 游戏结束标志
        self.winner = 0  # 赢家编号，0表示平局
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self):
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.uint8
        )  # 重置棋盘为空
        self.current_player = 1  # 重置为玩家1先手
        self.game_over = False  # 游戏未结束
        self.winner = 0  # 重置赢家编号
        return self.board

    def step(self, action):
        row, col = action
        if self.game_over:
            raise ValueError("Game is over. No more moves allowed.")

        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move: {(row, col)}")

        self.board[row, col] = self.current_player
        done = self.check_winner(row, col)
        if done:
            self.winner = self.current_player  # 记录赢家编号
            self.game_over = True
        elif self.is_board_full():  # 检查棋盘是否已满
            self.game_over = True  # 宣布和棋
        else:
            self.current_player = 3 - self.current_player  # 切换玩家
        return self.board, 0, self.game_over, {}

    def check_winner(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == self.current_player
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == self.current_player
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def is_board_full(self):  # 检查棋盘是否已满
        return np.all(self.board != 0)

    def render(self, mode=None):
        if mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.board_size * 30, self.board_size * 30 + 50)
            )  # 增加底部空间
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)  # 创建字体对象

        self.screen.fill((200, 200, 200))  # 灰色背景
        # 绘制棋盘网格
        for i in range(self.board_size):
            for j in range(self.board_size):
                pygame.draw.rect(
                    self.screen, (0, 0, 0), (j * 30, i * 30, 30, 30), 1
                )  # 绘制网格线
                if self.board[i, j] == 1:
                    pygame.draw.circle(
                        self.screen, (0, 0, 0), (j * 30 + 15, i * 30 + 15), 10
                    )  # 黑棋
                elif self.board[i, j] == 2:
                    pygame.draw.circle(
                        self.screen, (255, 255, 255), (j * 30 + 15, i * 30 + 15), 10
                    )  # 白棋

        # 动态生成结果文本
        result_text = ""
        if self.game_over:
            if self.winner == 1:
                result_text = "Player 1 wins!"
            elif self.winner == 2:
                result_text = "Player 2 wins!"
            else:
                result_text = "It's a draw!"

        if result_text:
            text_surface = self.font.render(result_text, True, (0, 0, 0))
            self.screen.blit(
                text_surface, (10, self.board_size * 30 + 10)
            )  # 显示在底部

        pygame.display.flip()
