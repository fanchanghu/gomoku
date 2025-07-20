from .framework import Trajectory, DataSet
import torch.nn.functional as F
import torch


def simplest_policy_gradient(
    trajectories: list[Trajectory],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad()

    states = []
    actions = []
    returns = []
    for traj in trajectories:
        R = sum(sar.reward for sar in traj)
        for sar in traj:
            states.append(torch.as_tensor(sar.state))
            actions.append(sar.action)
            returns.append(R)

    states = torch.stack(states)  # [N, H, W]
    actions = torch.tensor(actions, dtype=torch.long)  # [N]
    returns = torch.tensor(returns, dtype=torch.float32)  # [N]

    device = next(policy.parameters()).device
    states = states.to(device)
    actions = actions.to(device)
    returns = returns.to(device)

    logits = policy(states)  # [N, num_actions]
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_act = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [N]

    batch_size = len(trajectories)
    loss = -(log_probs_act * returns).sum() / batch_size
    loss.backward()
    optimizer.step()
