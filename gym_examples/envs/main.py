import gymnasium as gym
import gym_examples
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# Tạo môi trường với render_mode="human"
env = gym.make('gym_examples/GridWorld-v0', render_mode="human")

# check_env(env)

# Tạo mô hình
model = PPO("MultiInputPolicy", env, verbose=1)

obs, info = env.reset()
n_steps = 100
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, info = env.reset()
