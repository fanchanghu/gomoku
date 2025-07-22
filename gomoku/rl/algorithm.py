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


def compute_gae(
    trajectory: Trajectory,
    value_net: torch.nn.Module,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    trajectory: Trajectory（List[SAR]），每个SAR有 state, action, reward, next_state, done
    value_net: 状态价值网络
    返回：优势（tensor），回报（tensor）
    """
    device = next(value_net.parameters()).device

    states = [torch.as_tensor(sar.state, device=device) for sar in trajectory]
    rewards = [sar.reward for sar in trajectory]
    next_states = [torch.as_tensor(sar.next_state, device=device) for sar in trajectory]
    dones = [sar.done for sar in trajectory]

    values = value_net(torch.stack(states)).detach().cpu().numpy()
    next_values = value_net(torch.stack(next_states)).detach().cpu().numpy()

    advantages = []
    gae = 0
    returns = []
    for t in reversed(range(len(trajectory))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (1 - dones[t])
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def optimize_policy(states, actions, advantages, policy, optimizer):
    device = next(policy.parameters()).device
    states = states.to(device)
    actions = actions.to(device)
    advantages = advantages.to(device)

    logits = policy(states)
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_act = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    batch_size = len(states)
    loss = -(log_probs_act * advantages).sum() / batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_value(states, returns, value_net, value_optimizer):
    device = next(value_net.parameters()).device
    states = states.to(device)
    returns = returns.to(device)

    values_pred = value_net(states).squeeze(-1)
    value_loss = F.mse_loss(values_pred, returns)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()


def vanilla_policy_gradient(
    trajectories: list[Trajectory],
    policy: torch.nn.Module,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> None:
    states = []
    actions = []
    advantages = []
    returns = []
    for traj in trajectories:
        traj_adv, traj_ret = compute_gae(traj, value_net, gamma, lam)
        for idx, sar in enumerate(traj):
            states.append(torch.as_tensor(sar.state))
            actions.append(sar.action)
            advantages.append(traj_adv[idx])
            returns.append(traj_ret[idx])

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    optimize_policy(states, actions, advantages, policy, optimizer)
    optimize_value(states, returns, value_net, value_optimizer)
