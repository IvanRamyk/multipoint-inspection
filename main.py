import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Waypoints-v4", render_mode="human", num_targets=5)
obs, info = env.reset()
for _ in range(500):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
env.close()
