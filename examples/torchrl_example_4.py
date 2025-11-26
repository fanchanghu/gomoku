import tempfile
import torch

from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.envs.utils import RandomPolicy

torch.manual_seed(0)

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)

for data in collector:
    print(data)
    break

print(data["collector", "traj_ids"])

from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer

buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=1000)
)

indices = buffer.extend(data)

assert len(buffer) == collector.frames_per_batch

sample = buffer.sample(batch_size=30)
print(sample)
