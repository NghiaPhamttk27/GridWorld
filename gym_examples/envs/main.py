import gymnasium
import gym_examples

try:
    env = gymnasium.make('gym_examples/GridWorld-v0')
    print("Môi trường đã được đăng ký.")
except gymnasium.error.UnregisteredEnv:
    print("Môi trường chưa được đăng ký.")
