import gymnasium as gym
import gym_examples
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# Tạo môi trường với render_mode="human"
env = gym.make('gym_examples/GridWorld-v0', render_mode="human")

check_env(env)

model_dir = "models"
model_name = "my_model"
model_path = f"{model_dir}/{model_name}"

log_dir = "logs"

# Tạo mô hình
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
# model = PPO.load(model_path, env)

TIMESTEPS = 5

model.learn(total_timesteps=1, reset_num_timesteps=True)
model.save(f"{model_path}")


# obs, info = env.reset()
# n_steps = 1
# for _ in range(n_steps):
#     # Random action
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     if done:
#         model.save(f"{model_path}")
#         obs, info = env.reset()
