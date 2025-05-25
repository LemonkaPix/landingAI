# 🧠 landingAI — DQN Agent for LunarLander-v3

This repository contains an implementation of a **Deep Q-Network (DQN)** agent trained to solve the [LunarLander-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from OpenAI Gymnasium.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/powered%20by-PyTorch-EE4C2C?logo=pytorch)
![License](https://img.shields.io/github/license/LemonkaPix/landingAI)

---

## 🚀 Quick Start

### Requirements

Install dependencies using pip:

```bash
pip install torch gymnasium numpy
```

### Running the Agent

```bash
python main.py
```

The agent will train for **1000 episodes** using the `LunarLander-v3` environment with live rendering enabled (`render_mode="human"`).

---

## 📂 Project Structure

- `main.py` — main training script for the DQN agent
- Model checkpoint saving/loading
- Reward, epsilon, and loss tracking

---

## 🧠 Agent Overview

- Fully connected neural network with one hidden layer
- ε-greedy policy for exploration
- Experience replay with a memory buffer of 1000 steps
- Mini-batch training using Mean Squared Error loss
- Adjustable hardware acceleration (MPS/CUDA/CPU)

---

## 💾 Checkpoints

Models are automatically saved every 10 episodes as:

```
model_checkpoint_0.pth
model_checkpoint_10.pth
...
```

To load a checkpoint:

```python
episode = load_checkpoint(model, optimizer, 'model_checkpoint_10.pth')
```

---

## 📈 Sample Output

```
Episode: 10/1000  Reward: 183.25  Epsilon: 0.602  MSE: 0.537  Loss: 0.73
```

---

## 🔧 Possible Extensions

- Reward/loss visualization (e.g. Matplotlib)
- Target networks for stability
- Prioritized experience replay
- Multi-environment training

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ✨ Author

Made by [@LemonkaPix](https://github.com/LemonkaPix) with ❤️ using PyTorch and Gymnasium.
