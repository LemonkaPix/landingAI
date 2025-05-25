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


class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.nn.functional.leaky_relu(self.fc1(state), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        return self.fc3(x)


env = gym.make("LunarLander-v3", render_mode="human")

env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
