from .framework import Trajectory, DataSet
import torch.nn.functional as F
import torch


def simplest_policy_gradient(
    trajectories: list[Trajectory],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad()

    traj_size = len(trajectories)

    for traj in trajectories:
        # 整条轨迹的回报
        R = sum(sar.reward for sar in traj)

        log_prob_sum = 0.0

        for sar in traj:
            state = torch.as_tensor(sar.state).unsqueeze(0).unsqueeze(0)
            action = torch.as_tensor([sar.action]).unsqueeze(0)

            logits = policy(state)
            log_prob_sum += F.log_softmax(logits, dim=-1).gather(1, action).squeeze()

        # 一次性反向传播
        (-log_prob_sum * R / traj_size).backward()  # 负号把最大化变成最小化

    optimizer.step()
