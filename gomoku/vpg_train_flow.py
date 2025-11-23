from typing import Literal
import torch
import copy
from concurrent.futures import ThreadPoolExecutor
from gomoku.rl import TrainFlow, DataSet, Trajectory
from gomoku.rl.algorithm import vanilla_policy_gradient
from gomoku import GomokuEnv, GomokuNet
from gomoku.gomoku_play import *
import logging
import numpy as np
from gomoku.train_utils import find_latest_model, print_model, clear_old_models, save_model_with_limit


class VPGTrainFlow(TrainFlow):
    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def __init__(self, mode: Literal["init", "continue"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"use device: {self.device}")

        self.policy_net = GomokuNet().to(self.device)
        self.value_net = GomokuNet(mode="value").to(self.device)

        if mode == "continue":
            model_path, k = find_latest_model()
            value_model_path = model_path.replace("policy_net_", "value_net_") if model_path else None
            import os
            if model_path is None or not value_model_path or not os.path.exists(value_model_path):
                logging.error("No model found for continuation. Exiting.")
                raise FileNotFoundError("No model found for continuation.")
            logging.info(f"Continuing from model: {model_path}, k={k}")
            self.init_k = k
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.value_net.load_state_dict(torch.load(value_model_path, map_location=self.device))
        elif mode == "init":
            logging.info("Initializing training from scratch.")
            clear_old_models("model/*.pth")
            logging.info("Cleared old models.")
            self.init_k = 0

        self.env = GomokuEnv()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=3e-4)
        self.baseline_policy_net = copy.deepcopy(self.policy_net)
        self.executor = ThreadPoolExecutor(max_workers=10)

        print_model("policy_net", self.policy_net)
        print_model("value_net", self.value_net)

    def print_trajectory(self, traj: Trajectory):
        actions = [
            (sar.action // self.env.board_size, sar.action % self.env.board_size)
            for sar in traj
        ]
        reward = traj[-1].reward
        logging.info(f"Sample trajectory, actions: {actions}, reward: {reward}")

    def update_dataset(self, D: DataSet):
        temperature = 2.5 - (self.k / 10000) if self.k < 10000 else 1.5
        D.clear()
        self.policy_net.train(mode=False)
        futures = [
            self.executor.submit(self_play, self.policy_net, GomokuEnv(), temperature=temperature)
            for _ in range(10)
        ]
        for future in futures:
            tr1, tr2 = future.result()
            D.append(tr1)
            D.append(tr2)

        traj = D[torch.randint(0, len(D), (1,)).item()]
        self.print_trajectory(traj)

    def train_step(self, D: DataSet):
        self.policy_net.train()
        self.value_net.train()
        vanilla_policy_gradient(
            D,
            self.policy_net,
            self.value_net,
            self.optimizer,
            self.value_optimizer,
            gamma=0.99,
            lam=0.95,
        )

    def eval_step(self):
        self.policy_net.train(mode=False)
        self.baseline_policy_net.train(mode=False)

        # 获取策略熵和胜率
        ent1, _, win_rate = play_multimes(
            self.policy_net, self.baseline_policy_net, self.env, 5
        )

        # 记录指标
        if not hasattr(self, 'entropy_history'):
            self.entropy_history = []
        if not hasattr(self, 'win_rate_history'):
            self.win_rate_history = []

        self.entropy_history.append(ent1)
        self.win_rate_history.append(win_rate)

        # 打印当前指标
        logging.info(f"Entropy {ent1:.4f}, Win(rate) {win_rate:.4f}")

        # 可视化展示（简单文本图形）
        self._visualize_metrics()

    def _visualize_metrics(self):
        """简单的文本可视化展示"""
        import numpy as np

        # 创建固定长度的显示窗口
        window_size = 20
        if len(self.entropy_history) > window_size:
            entropy_window = self.entropy_history[-window_size:]
            win_rate_window = self.win_rate_history[-window_size:]
        else:
            entropy_window = self.entropy_history
            win_rate_window = self.win_rate_history

        # 创建熵的文本图形
        entropy_min, entropy_max = 0, np.log(225)  # 假设最大动作数为225
        entropy_range = entropy_max - entropy_min
        entropy_graph = ""
        for e in entropy_window:
            bar_length = int(20 * (e - entropy_min) / entropy_range)
            entropy_graph += f"[{'#' * bar_length}{' ' * (20-bar_length)}] {e:.3f}\n"

        # 创建胜率的文本图形
        win_rate_graph = ""
        for wr in win_rate_window:
            bar_length = int(20 * wr)
            win_rate_graph += f"[{'#' * bar_length}{' ' * (20-bar_length)}] {wr:.3f}\n"

        # 打印图形
        logging.debug(f"\n=== Metric Visualization ===\nEntropy Trend:\n{entropy_graph}\nWin Rate Trend:\n{win_rate_graph}")

    def save_model(self, k: int):
        save_model_with_limit({"policy_net": self.policy_net, "value_net": self.value_net}, k, keep=10)
        self.baseline_policy_net = copy.deepcopy(self.policy_net)
