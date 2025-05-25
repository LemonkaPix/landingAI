import gymnasium as gym
import torch
from torch import nn
from collections import deque
from torch import optim
import random
import numpy as np
import math

LEARNING_RATE = 0.001
BATCH_SIZE = 16

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

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size        #count of informations about enviroment
        self.action_size = action_size      #count of agent actions
        self.discount_factor = 0.99         #reward drop rate
        self.epsilion_greedy = 1.0          #initial randomness factor (1 = 100%)
        self.epsilion_greedy_min = 0.1      #minimal randomness factor
        self.epsilion_greedy_decay = 0.995  #decreasing randomness factor by 5%
        self.memory = deque(maxlen=1000)    #collection containing last 1000 events
        self.train_start = 500              #initial count of events

        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilion_greedy:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values_predicted = self.model(state)
        return torch.argmax(q_values_predicted).item()

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        data_batch = random.sample(self.memory, BATCH_SIZE)

        total_mse_loss = 0
        for state, action, reward, next_state, done in data_batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            discounted_reward = reward
            if not done:
                discounted_reward += self.discount_factor * torch.max(self.model(next_state))

            dqn_prediction = self.model(state)
            true_reward = dqn_prediction.clone()
            true_reward[action] = discounted_reward

            loss = self.criterion(dqn_prediction, true_reward)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_mse_loss += loss.item()

        if self.epsilion_greedy > self.epsilion_greedy_min:
            self.epsilion_greedy *= self.epsilion_greedy_decay

        return total_mse_loss / BATCH_SIZE

def save_checkpoint(model, optimizer, episode, filename="model_checkpoint.pth"):
    checkpoint = {
        'episode': episode,
        'model_state_disct': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    return episode

env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

rewards_history = []
epsilion_history = []
loss_history = []

episodes = 1000
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    total_reward = 0
    total_mse_loss = 0
    step_counter = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        step_counter += 1

        mse_loss = agent.replay()
        if mse_loss is not None:
            total_mse_loss += mse_loss

        if done:
            average_loss = total_mse_loss / step_counter if  step_counter > 0 else 0
            print(f'Episode: {episode+1}/{episodes}\tReward: {total_reward:.2f}\t'
                  f'Epsilion: {agent.epsilion_greedy:.3f}\tMSE: {total_mse_loss:.3f}\t'
                  f'Loss: {math.sqrt(total_mse_loss):.2f}')
            rewards_history.append(total_reward)
            epsilion_history.append(agent.epsilion_greedy)
            loss_history.append(math.sqrt(average_loss))
            break

    if episode % 10 == 0:
        save_checkpoint(agent.model, agent.optimizer,
                        episode, filename=f'model_checkpoint_{episode}.pth')
env.close()