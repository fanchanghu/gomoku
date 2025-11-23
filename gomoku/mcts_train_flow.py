"""
基于 MCTS 和 torchrl 的五子棋训练流程
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Literal, List, Dict
import copy
import random
from concurrent.futures import ThreadPoolExecutor

from gomoku import GomokuEnv, GomokuNet
from gomoku.mcts import MCTS
from gomoku.train_utils import find_latest_model, print_model, clear_old_models, save_model_with_limit


class SimpleReplayBuffer:
    """简单的经验回放缓冲区实现"""
    
    def __init__(self, capacity: int, batch_size: int = 32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0
        
    def add(self, data: Dict):
        """添加数据到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity
        
    def sample(self):
        """从缓冲区采样一个批次的数据"""
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"缓冲区数据不足: {len(self.buffer)} < {self.batch_size}")
            
        indices = random.sample(range(len(self.buffer)), self.batch_size)
        batch = {}
        
        # 收集所有键
        keys = self.buffer[0].keys()
        for key in keys:
            batch[key] = []
            
        # 填充批次数据
        for idx in indices:
            data = self.buffer[idx]
            for key in keys:
                batch[key].append(data[key])
                
        # 转换为张量
        for key in keys:
            batch[key] = torch.stack(batch[key])
            
        return batch
        
    def __len__(self):
        return len(self.buffer)


class MCTSTrainFlow:
    """基于 MCTS 的训练流程"""
    
    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def __init__(self, mode: Literal["init", "continue"], num_simulations: int = 800):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"使用设备: {self.device}")

        # 初始化策略网络
        self.policy_net = GomokuNet().to(self.device)
        self.value_net = GomokuNet(mode="value").to(self.device)

        if mode == "continue":
            # 继续训练模式
            model_path, k = find_latest_model()
            value_model_path = model_path.replace("policy_net_", "value_net_") if model_path else None
            import os
            if model_path is None or not value_model_path or not os.path.exists(value_model_path):
                logging.error("未找到用于继续训练的模型。退出。")
                raise FileNotFoundError("未找到用于继续训练的模型。")
            logging.info(f"从模型继续训练: {model_path}, k={k}")
            self.init_k = k
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.value_net.load_state_dict(torch.load(value_model_path, map_location=self.device))
        elif mode == "init":
            # 从头开始训练模式
            logging.info("从头开始初始化训练。")
            clear_old_models("model/*.pth")
            logging.info("已清理旧模型。")
            self.init_k = 0

        # 初始化环境
        self.env = GomokuEnv()
        
        # 初始化 MCTS
        self.mcts = MCTS(self.policy_net, self.env, num_simulations)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
        
        # 基线网络用于评估
        self.baseline_policy_net = copy.deepcopy(self.policy_net)
        
        # 线程池用于并行数据收集
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 经验回放缓冲区
        self.replay_buffer = None  # 延迟初始化
        self.replay_capacity = 10000
        
        # 打印模型信息
        print_model("policy_net", self.policy_net)
        print_model("value_net", self.value_net)

    def _create_replay_buffer(self, capacity: int):
        """创建简单的经验回放缓冲区"""
        return SimpleReplayBuffer(capacity)

    def collect_experience(self, num_games: int = 10):
        """收集经验数据"""
        self.policy_net.eval()
        
        # 延迟初始化经验回放缓冲区
        if self.replay_buffer is None:
            self.replay_buffer = self._create_replay_buffer(self.replay_capacity)
        
        futures = [
            self.executor.submit(self._play_one_game)
            for _ in range(num_games)
        ]
        
        for future in futures:
            game_data = future.result()
            if game_data:
                for data in game_data:
                    self.replay_buffer.add(data)

    def _play_one_game(self):
        """玩一局游戏并收集数据"""
        env = GomokuEnv()
        state = env.reset()
        game_data = []
        
        while not env.game_over:
            # 使用 MCTS 搜索获取策略和价值
            policy, value = self.mcts.search(state)
            
            # 根据策略选择动作
            valid_actions = np.argwhere(state == 0)
            if len(valid_actions) == 0:
                break
                
            # 将策略转换为动作概率
            action_probs = policy.flatten()
            valid_indices = [i * env.board_size + j for i, j in valid_actions]
            
            # 归一化有效动作的概率
            valid_probs = action_probs[valid_indices]
            if np.sum(valid_probs) > 0:
                valid_probs = valid_probs / np.sum(valid_probs)
            else:
                valid_probs = np.ones(len(valid_indices)) / len(valid_indices)
                
            # 根据概率选择动作
            action_idx = np.random.choice(valid_indices, p=valid_probs)
            row, col = divmod(action_idx, env.board_size)
            action = (row, col)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储转换数据
            data = {
                "state": torch.from_numpy(state.copy()).float(),
                "action": torch.tensor(action_idx).long(),
                "reward": torch.tensor(reward).float(),
                "next_state": torch.from_numpy(next_state.copy()).float(),
                "done": torch.tensor(done).bool(),
                "policy": torch.from_numpy(policy.flatten()).float(),
                "value": torch.tensor(value).float()
            }
            game_data.append(data)
            
            state = next_state
            
        return game_data

    def train_step(self):
        """执行训练步骤"""
        self.policy_net.train()
        self.value_net.train()
        
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            logging.info("经验回放缓冲区数据不足，跳过训练步骤")
            return
            
        # 从回放缓冲区采样
        batch = self.replay_buffer.sample()
        
        # 策略网络训练 - 最小化 KL 散度
        states = batch["state"].to(self.device)
        target_policies = batch["policy"].to(self.device)
        
        # 获取当前策略
        current_logits = self.policy_net(states)
        
        # 计算 KL 散度损失
        kl_loss = self._compute_kl_loss(current_logits, target_policies)
        
        # 价值网络训练 - 最小化价值误差
        target_values = batch["value"].to(self.device)
        predicted_values = self.value_net(states).squeeze(-1)
        value_loss = nn.MSELoss()(predicted_values, target_values)
        
        # 反向传播
        self.policy_optimizer.zero_grad()
        kl_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 记录损失
        if not hasattr(self, 'kl_loss_history'):
            self.kl_loss_history = []
            self.value_loss_history = []
            
        self.kl_loss_history.append(kl_loss.item())
        self.value_loss_history.append(value_loss.item())
        
        logging.info(f"训练步骤 - KL损失: {kl_loss.item():.4f}, 价值损失: {value_loss.item():.4f}")

    def _compute_kl_loss(self, current_logits, target_policies):
        """计算 KL 散度损失"""
        current_probs = torch.softmax(current_logits, dim=-1)
        target_probs = target_policies
        
        # 避免数值问题
        epsilon = 1e-8
        current_probs = torch.clamp(current_probs, epsilon, 1.0 - epsilon)
        target_probs = torch.clamp(target_probs, epsilon, 1.0 - epsilon)
        
        # 计算 KL 散度
        kl_div = target_probs * (torch.log(target_probs) - torch.log(current_probs))
        kl_loss = torch.sum(kl_div, dim=-1).mean()
        
        return kl_loss

    def eval_step(self):
        """评估步骤"""
        self.policy_net.eval()
        self.baseline_policy_net.eval()
        
        # 计算策略熵
        entropy = self._compute_policy_entropy()
        
        # 与基线网络对战评估胜率
        win_rate = self._evaluate_win_rate(num_games=5)
        
        # 记录指标
        if not hasattr(self, 'entropy_history'):
            self.entropy_history = []
        if not hasattr(self, 'win_rate_history'):
            self.win_rate_history = []
            
        self.entropy_history.append(entropy)
        self.win_rate_history.append(win_rate)
        
        logging.info(f"评估 - 策略熵: {entropy:.4f}, 胜率: {win_rate:.4f}")
        
        # 可视化指标
        self._visualize_metrics()

    def _compute_policy_entropy(self):
        """计算策略熵"""
        env = GomokuEnv()
        state = env.reset()
        
        # 使用 MCTS 获取策略
        policy, _ = self.mcts.search(state)
        
        # 计算熵
        valid_actions = np.argwhere(state == 0)
        valid_indices = [i * env.board_size + j for i, j in valid_actions]
        valid_probs = policy.flatten()[valid_indices]
        
        if len(valid_probs) > 0 and np.sum(valid_probs) > 0:
            valid_probs = valid_probs / np.sum(valid_probs)
            entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-8))
        else:
            entropy = 0.0
            
        return entropy

    def _evaluate_win_rate(self, num_games: int = 5):
        """评估与基线网络的胜率"""
        wins = 0
        
        for _ in range(num_games):
            env = GomokuEnv()
            state = env.reset()
            current_player = 1
            
            while not env.game_over:
                if current_player == 1:
                    # 当前策略网络
                    policy, _ = self.mcts.search(state)
                else:
                    # 基线网络
                    baseline_mcts = MCTS(self.baseline_policy_net, env, num_simulations=200)
                    policy, _ = baseline_mcts.search(state)
                
                # 选择动作
                valid_actions = np.argwhere(state == 0)
                if len(valid_actions) == 0:
                    break
                    
                action_probs = policy.flatten()
                valid_indices = [i * env.board_size + j for i, j in valid_actions]
                valid_probs = action_probs[valid_indices]
                
                if np.sum(valid_probs) > 0:
                    valid_probs = valid_probs / np.sum(valid_probs)
                    action_idx = np.random.choice(valid_indices, p=valid_probs)
                else:
                    action_idx = np.random.choice(valid_indices)
                    
                row, col = divmod(action_idx, env.board_size)
                state, _, done, _ = env.step((row, col))
                current_player = 3 - current_player
                
            # 统计胜率
            if env.winner == 1:  # 当前策略网络获胜
                wins += 1
                
        return wins / num_games

    def _visualize_metrics(self):
        """可视化训练指标"""
        import numpy as np
        
        # 创建固定长度的显示窗口
        window_size = 10
        if len(self.entropy_history) > window_size:
            entropy_window = self.entropy_history[-window_size:]
            win_rate_window = self.win_rate_history[-window_size:]
            kl_loss_window = self.kl_loss_history[-window_size:] if hasattr(self, 'kl_loss_history') else []
            value_loss_window = self.value_loss_history[-window_size:] if hasattr(self, 'value_loss_history') else []
        else:
            entropy_window = self.entropy_history
            win_rate_window = self.win_rate_history
            kl_loss_window = self.kl_loss_history if hasattr(self, 'kl_loss_history') else []
            value_loss_window = self.value_loss_history if hasattr(self, 'value_loss_history') else []
        
        # 创建文本图形
        logging.debug("\n=== 训练指标可视化 ===")
        
        if entropy_window:
            logging.debug(f"策略熵趋势: {entropy_window}")
        if win_rate_window:
            logging.debug(f"胜率趋势: {win_rate_window}")
        if kl_loss_window:
            logging.debug(f"KL损失趋势: {kl_loss_window}")
        if value_loss_window:
            logging.debug(f"价值损失趋势: {value_loss_window}")

    def save_model(self, k: int):
        """保存模型"""
        save_model_with_limit(
            {"policy_net": self.policy_net, "value_net": self.value_net}, 
            k, 
            keep=10
        )
        self.baseline_policy_net = copy.deepcopy(self.policy_net)
