"""
五子棋 MCTS 训练脚本
使用 torchrl 训练基于 MCTS 的五子棋对战算法
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# torchrl 相关导入
import torchrl
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, ToTensorImage, Resize, GrayScale
from torchrl.modules import (
    ActorValueOperator, 
    MLP, 
    ConvNet, 
    ProbabilisticActor, 
    ValueOperator,
    SafeModule
)
from tensordict.nn import TensorDictModule
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import A2CLoss, ClipPPOLoss, DDPGLoss
from torchrl.objectives.value import GAE
from torchrl.trainers import Recorder, RewardNormalizer

from torch.utils.tensorboard import SummaryWriter
from collections import deque

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入gomoku包以触发环境注册
from . import gym_registration

from gomoku.gomoku_env import GomokuEnv
from gomoku.gomoku_net import GomokuNet
from gomoku.mcts import MCTS
from gomoku.train_utils import (
    find_latest_model, 
    print_model, 
    clear_old_models, 
    save_model_with_limit
)


class GomokuTorchRLTrainer:
    """使用 torchrl 的五子棋训练器类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化环境
        self.env = self._create_env()
        
        # 初始化策略网络和价值网络
        self.policy_net = GomokuNet(
            board_size=args.board_size, 
            mode="policy"
        ).to(self.device)
        
        self.value_net = GomokuNet(
            board_size=args.board_size, 
            mode="value"
        ).to(self.device)
        
        # 创建 torchrl 策略和价值算子
        self.policy_module = self._create_policy_module()
        self.value_module = self._create_value_module()
        
        # 创建 Actor-Critic 模块
        # 使用独立的策略和价值网络，不使用共享模块
        self.actor_critic = ActorValueOperator(
            common_operator=None,
            policy_operator=self.policy_module,
            value_operator=self.value_module
        )
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 初始化 MCTS
        self.mcts = MCTS(
            policy_net=self.policy_net,
            env=self.env,
            num_simulations=args.num_simulations
        )
        
        # 创建损失函数
        self.loss_module = self._create_loss_module()
        
        # 创建数据收集器
        self.collector = self._create_collector()
        
        # 创建回放缓冲区
        self.replay_buffer = self._create_replay_buffer()
        
        # 训练统计
        self.episode = 0
        self.best_win_rate = 0.0
        self.win_rates = deque(maxlen=100)
        
        # TensorBoard 记录器
        self.writer = SummaryWriter(log_dir=args.log_dir)
        
        # 创建必要的目录
        os.makedirs("model", exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _create_env(self):
        """创建 torchrl 环境"""
        # 创建自定义的 Gym 环境包装器
        env = GymEnv("GomokuEnv-15x15", device=self.device)
        
        # 添加环境变换
        transforms = Compose(
            ToTensorImage(in_keys=["pixels"], out_keys=["pixels"]),
            Resize(self.args.board_size, self.args.board_size, in_keys=["pixels"]),
            GrayScale(in_keys=["pixels"])
        )
        
        return TransformedEnv(env, transforms)
        
    def _create_policy_module(self):
        """创建策略模块"""
        # 使用 TensorDictModule 包装策略网络
        policy_module = TensorDictModule(
            module=self.policy_net,
            in_keys=["pixels"],
            out_keys=["logits"]
        )
        
        return ProbabilisticActor(
            module=policy_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True
        )
        
    def _create_value_module(self):
        """创建价值模块"""
        return ValueOperator(
            module=self.value_net,
            in_keys=["pixels"],
            out_keys=["state_value"]
        )
        
    def _create_loss_module(self):
        """创建损失模块"""
        # 使用 PPO 损失
        advantage_module = GAE(
            gamma=self.args.gamma,
            lmbda=self.args.lmbda,
            value_network=self.value_module,
            average_gae=True
        )
        
        return ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            advantage_module=advantage_module,
            clip_epsilon=self.args.clip_epsilon,
            entropy_coef=self.args.entropy_coef,
            value_coef=self.args.value_coef,
            loss_critic_type="smooth_l1"
        )
        
    def _create_collector(self):
        """创建数据收集器"""
        return SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.args.frames_per_batch,
            total_frames=self.args.total_frames,
            device=self.device
        )
        
    def _create_replay_buffer(self):
        """创建回放缓冲区"""
        storage = LazyMemmapStorage(
            max_size=self.args.buffer_size,
            scratch_dir="./workdir/rb_cache",
            device=self.device
        )
        
        return ReplayBuffer(storage=storage)
        
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.args.log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        
    def load_checkpoint(self):
        """加载检查点"""
        policy_file, k = find_latest_model("policy_net_")
        value_file, _ = find_latest_model("value_net_")
        
        if policy_file and os.path.exists(policy_file):
            self.policy_net.load_state_dict(torch.load(policy_file, map_location=self.device))
            logging.info(f"Loaded policy network from {policy_file}")
            
        if value_file and os.path.exists(value_file):
            self.value_net.load_state_dict(torch.load(value_file, map_location=self.device))
            logging.info(f"Loaded value network from {value_file}")
            
        self.episode = k
        logging.info(f"Resuming from episode {self.episode}")
        
    def save_checkpoint(self):
        """保存检查点"""
        model_dict = {
            "policy_net": self.policy_net,
            "value_net": self.value_net
        }
        save_model_with_limit(model_dict, self.episode, keep=self.args.keep_models)
        
        # 保存优化器状态
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'best_win_rate': self.best_win_rate
        }, f"model/trainer_state_{self.episode+1}.pth")
        
    def collect_data(self):
        """收集训练数据"""
        # 使用数据收集器收集数据
        data = self.collector.next()
        
        # 将数据添加到回放缓冲区
        self.replay_buffer.extend(data)
        
        return data
        
    def train_step(self, data):
        """执行一步训练"""
        # 从回放缓冲区采样
        sampled_data = self.replay_buffer.sample(self.args.batch_size)
        
        # 计算损失
        loss_dict = self.loss_module(sampled_data)
        loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "policy_loss": loss_dict["loss_objective"].item(),
            "value_loss": loss_dict["loss_critic"].item(),
            "entropy_loss": loss_dict["loss_entropy"].item()
        }
        
    def evaluate(self, num_games=100):
        """评估当前模型性能"""
        wins = 0
        draws = 0
        losses = 0
        
        for _ in range(num_games):
            state = self.env.reset()
            done = False
            
            while not done:
                # 使用当前策略网络进行游戏
                with torch.no_grad():
                    action = self.policy_module(state)
                    
                state, reward, done, info = self.env.step(action)
                
            # 统计结果
            if info.get("winner", 0) == 1:  # 黑棋获胜（当前策略）
                wins += 1
            elif info.get("winner", 0) == 0:  # 平局
                draws += 1
            else:  # 白棋获胜
                losses += 1
                
        win_rate = wins / num_games
        self.win_rates.append(win_rate)
        
        return win_rate, wins, draws, losses
        
    def train(self):
        """主训练循环"""
        logging.info("Starting training with torchrl...")
        print_model("Policy Network", self.policy_net)
        print_model("Value Network", self.value_net)
        
        # 加载检查点（如果存在）
        if self.args.resume:
            self.load_checkpoint()
            
        start_time = time.time()
        
        for episode in range(self.episode, self.args.num_episodes):
            # 收集数据
            data = self.collect_data()
            
            # 训练网络
            loss_dict = self.train_step(data)
            
            # 记录训练统计
            self.writer.add_scalar("Loss/Total", loss_dict["total_loss"], episode)
            self.writer.add_scalar("Loss/Policy", loss_dict["policy_loss"], episode)
            self.writer.add_scalar("Loss/Value", loss_dict["value_loss"], episode)
            self.writer.add_scalar("Loss/Entropy", loss_dict["entropy_loss"], episode)
            
            # 定期评估
            if episode % self.args.eval_interval == 0:
                win_rate, wins, draws, losses = self.evaluate(num_games=self.args.eval_games)
                avg_win_rate = np.mean(self.win_rates) if self.win_rates else win_rate
                
                self.writer.add_scalar("Metrics/WinRate", win_rate, episode)
                self.writer.add_scalar("Metrics/AvgWinRate", avg_win_rate, episode)
                
                logging.info(
                    f"Episode {episode}: "
                    f"WinRate={win_rate:.3f}, "
                    f"AvgWinRate={avg_win_rate:.3f}, "
                    f"W/D/L={wins}/{draws}/{losses}, "
                    f"TotalLoss={loss_dict['total_loss']:.4f}, "
                    f"PolicyLoss={loss_dict['policy_loss']:.4f}, "
                    f"ValueLoss={loss_dict['value_loss']:.4f}"
                )
                
                # 保存最佳模型
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    torch.save(
                        self.policy_net.state_dict(), 
                        f"model/best_policy_net.pth"
                    )
                    torch.save(
                        self.value_net.state_dict(), 
                        f"model/best_value_net.pth"
                    )
                    logging.info(f"New best model saved with win rate: {win_rate:.3f}")
                    
            # 定期保存检查点
            if episode % self.args.save_interval == 0:
                self.save_checkpoint()
                logging.info(f"Checkpoint saved at episode {episode}")
                
            # 更新进度
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (episode - self.episode + 1) * (self.args.num_episodes - episode)
                logging.info(
                    f"Progress: {episode}/{self.args.num_episodes} "
                    f"({episode/self.args.num_episodes*100:.1f}%), "
                    f"ETA: {eta/3600:.1f}h"
                )
                
        # 训练完成
        self.save_checkpoint()
        self.writer.close()
        logging.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="五子棋 MCTS 训练脚本 (使用 torchrl)")
    
    # 训练参数
    parser.add_argument("--num-episodes", type=int, default=10000, help="训练回合数")
    parser.add_argument("--num-simulations", type=int, default=800, help="MCTS 模拟次数")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="梯度裁剪")
    
    # PPO 参数
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda 参数")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip 参数")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵系数")
    parser.add_argument("--value-coef", type=float, default=0.5, help="价值损失系数")
    
    # 数据收集参数
    parser.add_argument("--frames-per-batch", type=int, default=1000, help="每批次帧数")
    parser.add_argument("--total-frames", type=int, default=1000000, help="总帧数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--buffer-size", type=int, default=100000, help="回放缓冲区大小")
    
    # 环境参数
    parser.add_argument("--board-size", type=int, default=15, help="棋盘大小")
    
    # 评估参数
    parser.add_argument("--eval-interval", type=int, default=100, help="评估间隔")
    parser.add_argument("--eval-games", type=int, default=50, help="评估游戏数量")
    
    # 保存参数
    parser.add_argument("--save-interval", type=int, default=500, help="保存间隔")
    parser.add_argument("--keep-models", type=int, default=10, help="保留的模型数量")
    
    # 探索参数
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet 噪声参数")
    parser.add_argument("--noise-frac", type=float, default=0.25, help="噪声比例")
    
    # 其他参数
    parser.add_argument("--log-dir", type=str, default="./workdir/logs", help="日志目录")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = GomokuTorchRLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
