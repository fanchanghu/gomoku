from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .gomoku_env import GomokuEnv
from .rl import StateActionReward, Trajectory


def canonical_state(board: np.ndarray, cur_player: int) -> np.ndarray:
    """
    返回当前玩家视角的棋盘：
    当前玩家子 ->  1
    对手子     -> -1
    空位       ->  0
    """
    out = np.zeros_like(board, dtype=np.float32)
    out[board == cur_player] = 1.0
    out[board == 3 - cur_player] = -1.0
    return out


@torch.no_grad()
def select_action(
    policy_net: torch.nn.Module,
    env: GomokuEnv,
    state: np.ndarray,
) -> int:
    """
    根据策略网络在合法动作中做 softmax 采样。
    state      : (1, H, W) 当前玩家视角
    返回扁平化动作索引。
    """
    valid_mask = (env.board == 0).flatten()
    input = torch.from_numpy(state).reshape((1, 1, *state.shape))
    logits = policy_net(input).squeeze(0)  # (H*W,)
    logits[~valid_mask] = -torch.inf
    probs = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probs, 1).item())


def self_play(
    policy_net: torch.nn.Module,
    env: GomokuEnv,
) -> Tuple[Trajectory, Trajectory]:
    env.reset()
    board_size = env.board_size

    # 立即开始收集轨迹
    traj1: Trajectory = []
    traj2: Trajectory = []

    while not env.game_over:
        cur_p = env.current_player

        state = canonical_state(env.board, cur_p)
        act_idx = select_action(policy_net, env, state)

        # 当前步骤奖励先填 0
        sar = StateActionReward(state, act_idx, 0.0)

        if cur_p == 1:
            traj1.append(sar)
        else:
            traj2.append(sar)

        env.step((act_idx // board_size, act_idx % board_size))

    # 终局奖励
    if env.winner == 0:
        final_r = 0.0
    else:
        final_r = 1.0 if env.winner == 1 else -1.0

    # 只修改最后一步的奖励，且由于 sar 是不可变对象，需重新创建
    if traj1:
        traj1[-1] = StateActionReward(traj1[-1].state, traj1[-1].action, final_r)
    if traj2:
        traj2[-1] = StateActionReward(traj2[-1].state, traj2[-1].action, -final_r)

    return traj1, traj2


def play(
    policy_net: torch.nn.Module,
    policy_net2: torch.nn.Module,
    env: GomokuEnv,
) -> Tuple[float, float, int]:
    """
    双方依次落子，直至终局
    policy_net 先手，policy_net2 后手

    返回:
        entropy_net1: 先手策略在整个对局中的平均熵
        entropy_net2: 后手策略在整个对局中的平均熵
        winner      : 1 表示先手胜, 2 表示后手胜, 0 表示平局
    """
    env.reset()
    board_size = env.board_size

    entropy_sum = {1: 0.0, 2: 0.0}  # 分别累加两个网络的熵
    step_cnt = {1: 0, 2: 0}  # 两个网络各走了多少步

    while not env.game_over:
        cur_p = env.current_player
        net = policy_net if cur_p == 1 else policy_net2

        # 构建输入
        state = canonical_state(env.board, cur_p)  # (H, W)
        state_t = torch.from_numpy(state).reshape((1, 1, *state.shape))  # (1, H, W)
        valid = (env.board == 0).flatten()  # (H*W,)

        with torch.no_grad():
            logits = net(state_t).squeeze(0)  # (H*W,)
            logits[~valid] = -float("inf")
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            # 只在合法动作上计算熵，排除 -inf 项
            valid_log_probs = log_probs[valid]
            valid_probs = probs[valid]
            entropy = -(valid_probs * valid_log_probs).sum().item()

            # 选动作（贪心或采样均可；这里用贪心保证确定性）
            action = int(torch.argmax(log_probs).item())

        entropy_sum[cur_p] += entropy
        step_cnt[cur_p] += 1

        # 落子
        env.step((action // board_size, action % board_size))

    # 计算平均熵
    entropy_net1 = entropy_sum[1] / max(step_cnt[1], 1)
    entropy_net2 = entropy_sum[2] / max(step_cnt[2], 1)
    winner = env.winner  # 1 先手胜, 2 后手胜, 0 平局

    return entropy_net1, entropy_net2, winner


import numpy as np
from typing import Tuple


def play_multimes(
    policy_net: torch.nn.Module,
    policy_net2: torch.nn.Module,
    env: GomokuEnv,
    times: int,
) -> Tuple[float, float, float]:
    """
    共对战 2*times 局：
        前 times 局：policy_net 先手
        后 times 局：policy_net2 先手
    返回：
        entropy_net1  每局平均熵（按局求平均）
        entropy_net2  每局平均熵（按局求平均）
        policy_net    总胜率（0~1）
    """
    ent1_list, ent2_list, win_list = [], [], []

    for game in range(2 * times):
        # 偶数：policy_net 先手；奇数：policy_net2 先手
        if game % 2 == 0:
            ent1, ent2, winner = play(policy_net, policy_net2, env)
            net_win = 1 if winner == 1 else 0  # policy_net 先手赢
        else:
            ent2, ent1, winner = play(policy_net2, policy_net, env)
            net_win = 1 if winner == 2 else 0  # policy_net 后手赢

        print(f"Game {game + 1}/{2 * times}, ent1: {ent1:.4f}, ent2: {ent2:.4f}, win: {net_win}")

        ent1_list.append(ent1)
        ent2_list.append(ent2)
        win_list.append(net_win)

    return (
        float(np.mean(ent1_list)),
        float(np.mean(ent2_list)),
        float(np.mean(win_list)),
    )
