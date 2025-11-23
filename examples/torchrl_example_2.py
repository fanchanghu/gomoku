import torch

from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

# policy 1: construction using TensorDictModule
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)

# policy 2: construction using Actor
from torchrl.modules import Actor

policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)

# policy 3: construction using MLP
from torchrl.modules import MLP

module = MLP(
    out_features=env.action_spec.shape[-1],
    num_cells=[32, 64],
    activation_class=torch.nn.Tanh,
)
policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)

# policy 4: construction of a probabilistic actor
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor

backbone = MLP(in_features=3, out_features=2)
extractor = NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
    td_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
    return_log_prob=True,
)

rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)

from torchrl.envs.utils import ExplorationType, set_exploration_type

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # takes the mean as action
    rollout = env.rollout(max_steps=10, policy=policy)
with set_exploration_type(ExplorationType.RANDOM):
    # Samples actions according to the dist
    rollout = env.rollout(max_steps=10, policy=policy)

# policy 5: construction of an epsilon-greedy policy
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule

policy = Actor(MLP(3, 1, num_cells=[32, 64]))
exploration_module = EGreedyModule(
    spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5
)

exploration_policy = TensorDictSequential(policy, exploration_module)

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # Turns off exploration
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
with set_exploration_type(ExplorationType.RANDOM):
    # Turns on exploration
    rollout = env.rollout(max_steps=10, policy=exploration_policy)

###################### Q-Value actors ##########################

print(
    "=" * 10,
    "Q-Value based actors",
    "=" * 10,
)

env = GymEnv("CartPole-v1")
print(env.action_spec)

num_actions = 2
value_net = TensorDictModule(
    MLP(out_features=num_actions, num_cells=[32, 32]),
    in_keys=["observation"],
    out_keys=["action_value"],
)

from torchrl.modules import QValueModule

policy = TensorDictSequential(
    value_net,  # writes action values in our tensordict
    QValueModule(spec=env.action_spec),  # Reads the "action_value" entry by default
)

rollout = env.rollout(max_steps=3, policy=policy)
print(rollout)

policy_explore = TensorDictSequential(policy, EGreedyModule(env.action_spec))

with set_exploration_type(ExplorationType.RANDOM):
    rollout_explore = env.rollout(max_steps=3, policy=policy_explore)
