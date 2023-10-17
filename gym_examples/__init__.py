from gymnasium.envs.registration import register
from gym_examples.envs.grid_world import GridWorldEnv

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
    max_episode_steps=300,
)
