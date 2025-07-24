from typing import Literal
import torch
import copy
from concurrent.futures import ThreadPoolExecutor
from gomoku.rl import TrainFlow, DataSet, Trajectory
from gomoku.rl.algorithm import simplest_policy_gradient
from gomoku import GomokuEnv, GomokuNet
from gomoku.gomoku_play import *
import logging


def find_latest_model():
    import os
    import re

    model_dir = "./model"
    pattern = re.compile(r"policy_net_(\d+)\.pth")
    latest_k = -1
    latest_file = None

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            k = int(match.group(1))
            if k > latest_k:
                latest_k = k
                latest_file = filename

    if latest_file is not None:
        return os.path.join(model_dir, latest_file), latest_k
    else:
        return None, 0


def print_model(name: str, model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"{name}, parameters: {total_params} \n{model}")


class SimplestPGTrainFlow(TrainFlow):
    def __del__(self):
        self.executor.shutdown()

    def __init__(self, mode: Literal["init", "continue"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"use device: {self.device}")

        if mode == "continue":
            model_path, k = find_latest_model()
            if model_path is None:
                logging.error("No model found for continuation. Exiting.")
                raise FileNotFoundError("No model found for continuation.")
            logging.info(f"Continuing from model: {model_path}, k={k}")
            self.init_k = k
            self.policy_net = GomokuNet().to(self.device)
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        elif mode == "init":
            logging.info("Initializing training from scratch.")
            # clear model/*.pth
            import os
            import glob
            for file in glob.glob("model/*.pth"):
                os.remove(file)
            logging.info("Cleared old models.")
            self.init_k = 0
            self.policy_net = GomokuNet().to(self.device)

        self.env = GomokuEnv()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.baseline_policy_net = copy.deepcopy(self.policy_net)
        self.executor = ThreadPoolExecutor(max_workers=10)

        print_model("policy_net", self.policy_net)

    def print_trajectory(self, traj: Trajectory):
        actions = [
            (sar.action // self.env.board_size, sar.action % self.env.board_size)
            for sar in traj
        ]
        reward = traj[-1].reward
        logging.info(f"Sample trajectory, actions: {actions}, reward: {reward}")

    def update_dataset(self, D: DataSet):
        D.clear()
        self.policy_net.train(mode=False)
        futures = [
            self.executor.submit(self_play, self.policy_net, GomokuEnv())
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
        simplest_policy_gradient(D, self.policy_net, self.optimizer)

    def eval_step(self):
        self.policy_net.train(mode=False)
        self.baseline_policy_net.train(mode=False)
        ent1, _, win_rate = play_multimes(
            self.policy_net, self.baseline_policy_net, self.env, 5
        )
        logging.info(f"Entropy {ent1:.4f}, Win(rate) {win_rate:.4f}")

    def save_model(self, k: int):
        # only keep the latest 10 models
        import os
        import re
        import glob
        for file in glob.glob("model/policy_net_*.pth"):
            k_match = re.search(r"policy_net_(\d+)\.pth", file)
            if k_match and int(k_match.group(1)) < k - 900:
                os.remove(file)
                logging.info(f"Removed old model: {file}")

        torch.save(self.policy_net.state_dict(), f"model/policy_net_{k+1}.pth")
        logging.info(f"Model saved to policy_net_{k+1}.pth")

        self.baseline_policy_net = copy.deepcopy(self.policy_net)
