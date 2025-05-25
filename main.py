import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()