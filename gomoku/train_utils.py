import os
import re
import glob
import torch
import logging

def find_latest_model(prefix="policy_net_"):
    model_dir = "./model"
    pattern = re.compile(rf"{prefix}(\d+)\.pth")
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

def clear_old_models(pattern="model/*.pth"):
    for file in glob.glob(pattern):
        os.remove(file)
        logging.info(f"Removed old model: {file}")

def save_model_with_limit(model_dict, k, keep=10):
    # model_dict: {name: model}
    for name, model in model_dict.items():
        # 清理旧模型
        for file in glob.glob(f"model/{name}_*.pth"):
            k_match = re.search(rf"{name}_(\d+)\.pth", file)
            if k_match and int(k_match.group(1)) < k - keep * 10:
                os.remove(file)
                logging.info(f"Removed old {name} model: {file}")
        # 保存新模型
        torch.save(model.state_dict(), f"model/{name}_{k+1}.pth")
        logging.info(f"Model saved to {name}_{k+1}.pth")
