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
    trajectory,
    value_net,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    device = next(value_net.parameters()).device
    n = len(trajectory)

    # 预分配 Tensor 缓冲区
    states = torch.empty((n, *trajectory[0].state.shape), device=device)
    next_states = torch.empty_like(states)
    rewards = torch.empty(n, device=device)
    dones = torch.zeros(n, device=device)  # 先全部填 0
    dones[-1] = 1.0  # 最后一个元素 done=1

    # 一次性把数据搬进 Tensor
    for t, sar in enumerate(trajectory):
        states[t] = torch.as_tensor(sar.state, device=device)
        rewards[t] = float(sar.reward)

    next_states[:-1] = states[1:]  # 0 ~ n-2 的 next_state
    next_states[-1] = states[-1]  # 最后一个 next_state 用自身

    with torch.no_grad():
        values = value_net(states).squeeze(-1)  # (n,)
        next_values = value_net(next_states).squeeze(-1)  # (n,)

    # 预分配 advantages & returns
    advantages = torch.empty(n, device=device)
    returns = torch.empty(n, device=device)

    gae = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (1 - dones[t])
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


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
