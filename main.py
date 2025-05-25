import gymnasium as gym
import torch
from torch import nn


OPTIMIZE_WITH_HARDWARE = False
device = torch.device('cpu')
if OPTIMIZE_WITH_HARDWARE:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Selected device: MPS (Metal Performace Shaders)')
    elif torch.backends.cuda.is_flash_attention_available():
        device = torch.device('cuda')
        print('Selected device: GPU with CUDA support')
else:
    print('Selected device: CPU')





env = gym.make("LunarLander-v3", render_mode="human")

env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
