"""
五子棋 Gym 环境注册
用于 torchrl 环境包装
"""

from gymnasium.envs.registration import register

register(
    id="GomokuEnv-15x15",
    entry_point="gomoku.gomoku_env:GomokuEnv",
    max_episode_steps=225,  # 15x15 棋盘的最大步数
    kwargs={"board_size": 15}
)

register(
    id="GomokuEnv-20x20",
    entry_point="gomoku.gomoku_env:GomokuEnv", 
    max_episode_steps=400,  # 20x20 棋盘的最大步数
    kwargs={"board_size": 20}
)
