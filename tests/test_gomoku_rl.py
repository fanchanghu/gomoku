"""
测试 torchrl 功能 - pytest 版本
"""

import sys
import os
import pytest
import torch

# 注册自定义标记
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 注册 Gym 环境
import gomoku.gym_registration

import gymnasium as gym
from torchrl.envs import GymEnv


class TestTorchRL:
    """测试 torchrl 功能"""

    def test_gym_registration(self):
        """测试 Gym 环境注册"""
        env = gym.make("GomokuEnv-15x15")
        
        # 测试环境重置
        obs, info = env.reset()
        assert obs.shape == (15, 15), f"观察空间形状不正确: {obs.shape}"
        
        # 测试动作空间
        assert hasattr(env, 'action_space'), "环境缺少 action_space 属性"
        assert env.action_space.shape == (2,), f"动作空间形状不正确: {env.action_space.shape}"
        assert env.action_space.dtype == int, f"动作空间类型不正确: {env.action_space.dtype}"
        
        env.close()

@pytest.fixture
def gomoku_env():
    """提供 Gomoku 环境 fixture"""
    env = GymEnv("GomokuEnv-15x15")
    yield env
    env.close()


@pytest.fixture
def policy_net():
    """提供策略网络 fixture"""
    from gomoku.gomoku_net import GomokuNet
    net = GomokuNet(board_size=15, mode="policy")
    return net


def test_env_with_fixture(gomoku_env: GymEnv):
    """使用 fixture 测试环境"""
    init = gomoku_env.reset()
    print(init)
    
    rollout = gomoku_env.rollout(max_steps=10)
    print(rollout)


def test_network_with_fixture(policy_net: GymEnv):
    """使用 fixture 测试网络"""
    dummy_input = torch.randn(1, 15, 15)
    output = policy_net(dummy_input)
    assert output.shape == torch.Size([1, 225])


@pytest.mark.parametrize("board_size", [15])
def test_networks_different_sizes(board_size):
    """参数化测试不同棋盘大小的网络"""
    from gomoku.gomoku_net import GomokuNet
    
    policy_net = GomokuNet(board_size=board_size, mode="policy")
    dummy_input = torch.randn(1, board_size, board_size)
    output = policy_net(dummy_input)
    
    expected_shape = torch.Size([1, board_size * board_size])
    assert output.shape == expected_shape, \
        f"棋盘大小 {board_size} 的输出形状不正确: {output.shape}"


def test_mcts_extended():
    """MCTS 扩展测试"""
    from gomoku.mcts import MCTS
    from gomoku.gomoku_env import GomokuEnv
    from gomoku.gomoku_net import GomokuNet
    
    env = GomokuEnv(board_size=15)
    policy_net = GomokuNet(board_size=15, mode="policy")
    
    # 使用更多模拟次数
    mcts = MCTS(policy_net=policy_net, env=env, num_simulations=50)
    
    state, _ = env.reset()
    policy, value = mcts.search(state)
    
    assert policy.shape == (15, 15)
    assert isinstance(value, float)
    assert -1 <= value <= 1
