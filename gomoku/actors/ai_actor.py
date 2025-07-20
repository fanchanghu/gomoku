import torch
import numpy as np
from gomoku import GomokuEnv
from gomoku import GomokuNet


class AIActor:
    def __init__(self):
        self.policy_net = GomokuNet()

        # load latest(max k) model from model/policy_net_{k}.pth
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
            model_path = os.path.join(model_dir, latest_file)
            self.policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded model from {model_path}, k={latest_k}")
        else:
            # throw error
            raise FileNotFoundError("No model found in the model directory.")

    def __call__(self, env: GomokuEnv):
        empty_positions = np.argwhere(env.board == 0)
        if len(empty_positions) == 0:
            raise ValueError("No valid moves left.")

        # 构造当前玩家视角的输入
        cur_player = env.current_player
        board = env.board
        state = np.zeros_like(board, dtype=np.float32)
        state[board == cur_player] = 1.0
        state[board == 3 - cur_player] = -1.0
        state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

        with torch.no_grad():
            logits = self.policy_net(state).flatten()  # [H*W]
            valid_mask = (env.board == 0).flatten()
            logits[~valid_mask] = -float("inf")
            probs = torch.softmax(logits, dim=0)
            action_idx = int(torch.argmax(probs).item())
        row, col = divmod(action_idx, env.board_size)
        return int(row), int(col)
