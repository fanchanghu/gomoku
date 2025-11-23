from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

reset = env.reset()
print(reset)

reset_with_action = env.rand_action(reset)
print(reset_with_action)
print(reset_with_action["action"])

stepped_data = env.step(reset_with_action)
print(stepped_data)

from torchrl.envs import step_mdp

data = step_mdp(stepped_data)
print(data)
