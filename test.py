import gym
import d4rl
import os
from buffer import ReplayBuffer
import numpy as np

env = gym.make("walker2d-medium-v0")
dataset = d4rl.qlearning_dataset(env)
offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_dim=np.prod(env.action_space.shape),
        action_dtype=np.float32)
offline_buffer.load_dataset(dataset)
data_dict = offline_buffer.sample(1)
reward = data_dict['rewards']
print(reward, reward[0], reward[0][0])

# state = env.reset()
# action = env.action_space.sample()
# next_state, reward, done, _ = env.step(action)
# print({
#             "observations": state,
#             "actions": action,
#             "next_observations": next_state,
#             "terminals": done,
#             "rewards": reward
#         })
