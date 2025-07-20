import torch
import copy
from concurrent.futures import ThreadPoolExecutor
from gomoku.rl import TrainFlow, DataSet, Trajectory
from gomoku.rl.algorithm import simplest_policy_gradient
from gomoku import GomokuEnv, GomokuNet
from gomoku.gomoku_play import *
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"


def print_model(name: str, model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"{name}, parameters: {total_params} \n{model}")


class GomokuFlow(TrainFlow):
    def __del__(self):
        self.executor.shutdown()

    def __init__(self):
        self.policy_net = GomokuNet().to(device)
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
        torch.save(self.policy_net.state_dict(), f"model/policy_net_{k+1}.pth")
        logging.info(f"Model saved to policy_net_{k+1}.pth")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"use device: {device}")
    train_flow = GomokuFlow()
    train_flow.run(max_k=1000, eval_interval=10, save_interval=100)
