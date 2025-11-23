# gomoku_env.py
"""
五子棋游戏环境实现
基于 Gymnasium 框架，提供标准的强化学习环境接口
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.locals import *

class GomokuEnv(gym.Env):
    """五子棋游戏环境类"""
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=15):
        """
        初始化五子棋环境
        
        Args:
            board_size: 棋盘大小，默认为15x15
        """
        self.board_size = board_size
        
        # 动作空间：二维坐标 (行, 列)
        self.action_space = spaces.Box(
            low=0, high=board_size - 1, shape=(2,), dtype=int
        )
        
        # 观察空间：棋盘状态 (board_size x board_size)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.uint8
        )
        
        # 棋盘状态：0=空, 1=黑棋, 2=白棋
        self.board = np.zeros((board_size, board_size), dtype=np.uint8)
        
        # 当前玩家：1=黑棋(先手), 2=白棋
        self.current_player = 1
        
        # 游戏状态标志
        self.game_over = False
        self.winner = 0  # 0=平局, 1=黑棋胜, 2=白棋胜
        
        # Pygame 渲染相关属性
        self.screen = None
        self.clock = None
        self.font = None
        
        # AI 移动概率可视化
        self.top_move_probs = None

    def reset(self):
        """
        重置游戏环境到初始状态
        
        Returns:
            np.ndarray: 重置后的棋盘状态
        """
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.uint8
        )  # 重置棋盘为空
        self.current_player = 1  # 重置为玩家1先手
        self.game_over = False  # 游戏未结束
        self.winner = 0  # 重置赢家编号
        return self.board

    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 动作坐标 (行, 列)
            
        Returns:
            tuple: (observation, reward, done, info)
                - observation: 新的棋盘状态
                - reward: 奖励值 (当前实现为0)
                - done: 游戏是否结束
                - info: 额外信息字典
                
        Raises:
            ValueError: 如果游戏已结束或动作无效
        """
        row, col = action
        if self.game_over:
            raise ValueError("Game is over. No more moves allowed.")

        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move: {(row, col)}")

        # 在指定位置落子
        self.board[row, col] = self.current_player
        
        # 检查是否获胜
        done = self.check_winner(row, col)
        if done:
            self.winner = self.current_player  # 记录赢家编号
            self.game_over = True
        elif self.is_board_full():  # 检查棋盘是否已满
            self.game_over = True  # 宣布和棋
        else:
            # 切换玩家 (1->2, 2->1)
            self.current_player = 3 - self.current_player
            
        return self.board, 0, self.game_over, {}

    def check_winner(self, row, col):
        """
        检查当前玩家在指定位置落子后是否获胜
        
        Args:
            row: 落子的行坐标
            col: 落子的列坐标
            
        Returns:
            bool: 如果当前玩家获胜返回True，否则返回False
        """
        # 检查方向：水平、垂直、主对角线、副对角线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # 当前位置已经有一个棋子
            
            # 正向检查
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
                    
            # 反向检查
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
                    
            # 如果连续棋子数达到5个，则获胜
            if count >= 5:
                return True
                
        return False

    def is_board_full(self):
        """
        检查棋盘是否已满（所有位置都有棋子）
        
        Returns:
            bool: 如果棋盘已满返回True，否则返回False
        """
        return np.all(self.board != 0)

    def render(self, mode=None):
        """
        渲染游戏界面
        
        Args:
            mode: 渲染模式，目前只支持 "human" 模式
            
        Note:
            使用 Pygame 进行图形化渲染，显示棋盘状态和游戏结果
        """
        if mode != "human":
            return

        # 初始化 Pygame 显示
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.board_size * 30, self.board_size * 30 + 50)
            )  # 增加底部空间用于显示结果
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)  # 创建字体对象

        # 绘制背景
        self.screen.fill((200, 200, 200))  # 灰色背景
        
        # 绘制棋盘网格和棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 绘制网格线
                pygame.draw.rect(
                    self.screen, (0, 0, 0), (j * 30, i * 30, 30, 30), 1
                )
                
                # 绘制棋子
                if self.board[i, j] == 1:
                    pygame.draw.circle(
                        self.screen, (0, 0, 0), (j * 30 + 15, i * 30 + 15), 10
                    )  # 黑棋
                elif self.board[i, j] == 2:
                    pygame.draw.circle(
                        self.screen, (255, 255, 255), (j * 30 + 15, i * 30 + 15), 10
                    )  # 白棋

        # 绘制 AI 移动概率标注（用于调试和可视化）
        if self.top_move_probs is not None:
            board_size = self.board_size
            flat_probs = self.top_move_probs.flatten()
            top_n = min(10, np.count_nonzero(flat_probs))
            top_indices = np.argpartition(-flat_probs, top_n)[:top_n]
            top_indices = top_indices[np.argsort(-flat_probs[top_indices])]
            
            for idx in top_indices:
                row, col = divmod(idx, board_size)
                prob = flat_probs[idx]
                x, y = col * 30 + 15, row * 30 + 15
                
                # 绘制红色圆圈标注高概率位置
                pygame.draw.circle(self.screen, (255,0,0), (x, y), 10, 2)
                
                # 显示概率数值
                font = pygame.font.SysFont(None, 20)
                text = font.render(f"{prob:.2f}", True, (255,0,0))
                self.screen.blit(text, (x-10, y-10))

        # 显示游戏结果
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

        # 更新显示
        pygame.display.flip()
