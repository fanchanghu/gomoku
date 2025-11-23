"""
蒙特卡洛树搜索 (MCTS) 实现
用于五子棋游戏的强化学习训练
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Optional
from gomoku import GomokuEnv, GomokuNet


class MCTSNode:
    """MCTS 树节点"""
    
    def __init__(self, state: np.ndarray, parent=None, action=None, prior=0.0):
        self.state = state.copy()  # 棋盘状态
        self.parent = parent      # 父节点
        self.action = action      # 导致此节点的动作
        self.prior = prior       # 先验概率
        
        # 统计信息
        self.visit_count = 0
        self.value_sum = 0.0
        self.value = 0.0
        
        # 子节点
        self.children = {}
        
    @property
    def q_value(self) -> float:
        """Q值 = 平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    @property
    def u_value(self) -> float:
        """U值 = 探索价值"""
        if self.parent is None:
            return 0.0
        return (self.prior * math.sqrt(self.parent.visit_count) / 
                (1 + self.visit_count))
        
    @property
    def pucb_value(self) -> float:
        """PUCB值 = Q + U"""
        return self.q_value + self.u_value
        
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children) == 0
        
    def is_terminal(self, env: GomokuEnv) -> bool:
        """是否为终止状态"""
        # 检查游戏是否结束
        env.board = self.state
        env.game_over = False
        env.winner = 0
        
        # 检查棋盘是否已满
        if np.all(self.state != 0):
            return True
            
        # 检查是否有玩家获胜
        board_size = env.board_size
        for i in range(board_size):
            for j in range(board_size):
                if self.state[i, j] != 0:
                    # 检查四个方向
                    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                    for dr, dc in directions:
                        count = 1
                        # 正向检查
                        for k in range(1, 5):
                            r, c = i + dr * k, j + dc * k
                            if (0 <= r < board_size and 0 <= c < board_size and 
                                self.state[r, c] == self.state[i, j]):
                                count += 1
                            else:
                                break
                        # 反向检查
                        for k in range(1, 5):
                            r, c = i - dr * k, j - dc * k
                            if (0 <= r < board_size and 0 <= c < board_size and 
                                self.state[r, c] == self.state[i, j]):
                                count += 1
                            else:
                                break
                        if count >= 5:
                            return True
        return False
        
    def expand(self, policy_net: GomokuNet, env: GomokuEnv):
        """扩展节点"""
        if not self.is_leaf():
            return
            
        # 获取当前玩家
        current_player = self._get_current_player(env)
        
        # 使用策略网络获取动作概率
        state_tensor = self._state_to_tensor(self.state, current_player)
        with torch.no_grad():
            logits = policy_net(state_tensor).flatten()
            
        # 创建有效动作掩码
        valid_actions = np.argwhere(self.state == 0)
        valid_indices = [i * env.board_size + j for i, j in valid_actions]
        
        # 创建概率分布
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        
        # 为每个有效动作创建子节点
        for action_idx in valid_indices:
            row, col = divmod(action_idx, env.board_size)
            prior_prob = probs[action_idx]
            self.children[action_idx] = MCTSNode(
                state=self.state, 
                parent=self, 
                action=action_idx,
                prior=prior_prob
            )
            
    def _get_current_player(self, env: GomokuEnv) -> int:
        """从棋盘状态推断当前玩家"""
        # 统计黑白棋子数量
        black_count = np.sum(self.state == 1)
        white_count = np.sum(self.state == 2)
        
        # 黑棋先手，棋子数量相等时轮到黑棋
        if black_count == white_count:
            return 1  # 黑棋
        else:
            return 2  # 白棋
            
    def _state_to_tensor(self, state: np.ndarray, current_player: int) -> torch.Tensor:
        """将棋盘状态转换为神经网络输入张量"""
        # 创建当前玩家视角的输入
        board_tensor = np.zeros_like(state, dtype=np.float32)
        board_tensor[state == current_player] = 1.0
        board_tensor[state == 3 - current_player] = -1.0
        
        return torch.from_numpy(board_tensor).unsqueeze(0).unsqueeze(0).float()


class MCTS:
    """蒙特卡洛树搜索算法"""
    
    def __init__(self, policy_net: GomokuNet, env: GomokuEnv, num_simulations: int = 800):
        self.policy_net = policy_net
        self.env = env
        self.num_simulations = num_simulations
        
    def search(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        执行MCTS搜索
        
        Args:
            state: 当前棋盘状态
            
        Returns:
            policy: 动作概率分布
            value: 状态价值估计
        """
        root = MCTSNode(state)
        
        # 执行多次模拟
        for _ in range(self.num_simulations):
            self._simulate(root)
            
        # 根据访问次数计算策略
        policy = self._get_policy(root)
        value = root.q_value if root.visit_count > 0 else 0.0
        
        return policy, value
        
    def _simulate(self, root: MCTSNode):
        """执行一次模拟"""
        node = root
        env_copy = GomokuEnv(board_size=self.env.board_size)
        
        # 选择阶段
        while not node.is_leaf():
            if node.is_terminal(env_copy):
                break
                
            # 选择PUCB值最大的子节点
            best_child = None
            best_score = -float('inf')
            
            for child in node.children.values():
                score = child.pucb_value
                if score > best_score:
                    best_score = score
                    best_child = child
                    
            node = best_child
            
        # 扩展和评估阶段
        if not node.is_terminal(env_copy):
            node.expand(self.policy_net, env_copy)
            
            # 随机选择一个子节点进行rollout
            if node.children:
                child = list(node.children.values())[0]
                value = self._rollout(child.state, env_copy)
            else:
                value = 0.0
        else:
            # 终止状态的价值
            value = self._evaluate_terminal(node.state, env_copy)
            
        # 回溯更新
        self._backpropagate(node, value)
        
    def _rollout(self, state: np.ndarray, env: GomokuEnv) -> float:
        """执行随机rollout直到游戏结束"""
        env.board = state.copy()
        env.current_player = self._get_current_player(state, env)
        env.game_over = False
        env.winner = 0
        
        # 随机游戏直到结束
        while not env.game_over:
            # 获取所有合法动作
            valid_actions = np.argwhere(env.board == 0)
            if len(valid_actions) == 0:
                break
                
            # 随机选择动作
            action = valid_actions[np.random.choice(len(valid_actions))]
            _, _, terminated, truncated, _ = env.step(action)
            
        # 返回游戏结果
        return self._get_reward(env)
        
    def _evaluate_terminal(self, state: np.ndarray, env: GomokuEnv) -> float:
        """评估终止状态的价值"""
        env.board = state.copy()
        env.current_player = self._get_current_player(state, env)
        env.game_over = True
        
        # 检查获胜者
        board_size = env.board_size
        for i in range(board_size):
            for j in range(board_size):
                if state[i, j] != 0:
                    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                    for dr, dc in directions:
                        count = 1
                        for k in range(1, 5):
                            r, c = i + dr * k, j + dc * k
                            if (0 <= r < board_size and 0 <= c < board_size and 
                                state[r, c] == state[i, j]):
                                count += 1
                            else:
                                break
                        for k in range(1, 5):
                            r, c = i - dr * k, j - dc * k
                            if (0 <= r < board_size and 0 <= c < board_size and 
                                state[r, c] == state[i, j]):
                                count += 1
                            else:
                                break
                        if count >= 5:
                            env.winner = state[i, j]
                            return self._get_reward(env)
        
        # 平局
        return 0.0
        
    def _get_current_player(self, state: np.ndarray, env: GomokuEnv) -> int:
        """从棋盘状态推断当前玩家"""
        black_count = np.sum(state == 1)
        white_count = np.sum(state == 2)
        return 1 if black_count == white_count else 2
        
    def _get_reward(self, env: GomokuEnv) -> float:
        """获取奖励值"""
        if env.winner == 0:  # 平局
            return 0.0
        elif env.winner == env.current_player:  # 当前玩家获胜
            return 1.0
        else:  # 对手获胜
            return -1.0
            
    def _backpropagate(self, node: MCTSNode, value: float):
        """回溯更新节点统计信息"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node.value = node.value_sum / node.visit_count
            value = -value  # 交替视角
            node = node.parent
            
    def _get_policy(self, root: MCTSNode) -> np.ndarray:
        """根据访问次数计算策略分布"""
        policy = np.zeros(self.env.board_size * self.env.board_size)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits > 0:
            for action_idx, child in root.children.items():
                policy[action_idx] = child.visit_count / total_visits
                
        return policy.reshape(self.env.board_size, self.env.board_size)
