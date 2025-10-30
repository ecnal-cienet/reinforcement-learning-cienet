# Deep Q-Network (DQN) Implementation - CartPole-v1

> **中文版本**: [README_ch.md](README_ch.md)

## Overview

This is an implementation of the **Deep Q-Network (DQN)** algorithm solving OpenAI Gymnasium's **CartPole-v1** environment. DQN is the first algorithm to successfully combine deep learning with reinforcement learning, proposed by DeepMind in 2015, capable of handling high-dimensional state spaces.

**Core Breakthrough:** DQN uses a **deep neural network** as a function approximator to replace traditional Q-Tables, solving the "curse of dimensionality" problem of tabular methods.

## Environment Description

### CartPole-v1 (Inverted Pendulum)

CartPole is a classic control problem: a pole is attached through an unactuated joint to a cart. The goal is to move the cart left or right to keep the pole upright.

```
        |
        |  ← Pole
        |
    ┌───────┐
    │  Cart │ ← Can move left/right
    └───┬───┘
    ════════════
```

### Environment Parameters

- **State Space**: Continuous 4-dimensional vector
  - `position`: Cart position (range: -4.8 ~ 4.8)
  - `velocity`: Cart velocity
  - `angle`: Pole angle (range: -0.418 ~ 0.418 radians, approximately ±24°)
  - `angular_velocity`: Pole angular velocity

- **Action Space**: Discrete 2 actions
  - `0`: Push cart to the left
  - `1`: Push cart to the right

- **Reward Function**:
  - Receive `+1` reward for each timestep survived
  - Goal is to keep pole upright as long as possible

- **Termination Conditions**:
  - Pole tilt angle exceeds ±12°
  - Cart moves out of bounds
  - Reaches maximum steps (500 steps)

- **Success Criteria**:
  - Average reward over 100 consecutive episodes ≥ 475

## How to Run

### Prerequisites

Ensure you've activated the virtual environment and installed dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Execute the Program

```bash
python 2_Cart_Pole_DQN/cart_pole_dqn.py
```

Or from within the `2_Cart_Pole_DQN` directory:

```bash
cd 2_Cart_Pole_DQN
python cart_pole_dqn.py
```

## Algorithm Core

### Why Do We Need DQN?

**Q-Learning Problem:**
- CartPole's state space is **continuous** (e.g., position = 1.234567...)
- Cannot create a Q-Table for "infinitely" many states

**DQN Solution:**
- Use a **neural network** Q<sub>θ</sub>(s, a) to **estimate** Q-values
- Network can "generalize": similar states produce similar Q-values

### DQN's Two Key Stabilization Techniques

Directly training Q-Learning with a neural network is very unstable. DQN introduces two key techniques:

#### 1. Experience Replay

**Problem:** Neural network training hates "highly correlated" consecutive data, leading to overfitting.

**Solution:** Build a **Replay Buffer** (memory buffer)

```python
class ReplayBuffer:
    - Capacity (BUFFER_SIZE): 10,000 experiences
    - Storage format: (state, action, reward, next_state, done)
    - Training: randomly sample BATCH_SIZE=64 uncorrelated experiences
```

**Benefits:**
- ✅ Breaks temporal correlation in data
- ✅ Reuses past experiences (data efficient)
- ✅ More stable training

#### 2. Target Network

**Problem:** "Moving target" problem

In traditional Q-Learning, we use **the same network** to compute both "predicted value" and "target value":

```
Loss = [R + γ × max Q(S', a') - Q(S, A)]²
         └──────┬──────┘   └───┬───┘
              Target       Prediction
           (both from same network)
```

This is like shooting at a target **you control yourself** → you'll never catch up!

**Solution:** Use **two** neural networks

1. **Online Network** Q<sub>online</sub>
   - Role: Select actions, compute predicted values
   - Status: **Updates every step**

2. **Target Network** Q<sub>target</sub>
   - Role: Compute TD target values
   - Status: **Weights frozen** (synchronizes every 100 steps)

```python
# Compute TD Target (using Target Network - fixed target)
q_next_target = self.target_network(next_states)
td_target = rewards + GAMMA * jnp.max(q_next_target, axis=1)

# Compute Loss (using Online Network)
q_current = self.online_network(states)
loss = mean((q_current - td_target)²)
```

**Synchronization Mechanism:**
```python
if total_steps % TARGET_UPDATE_FREQ == 0:  # Every 100 steps
    agent.update_target_network()  # Copy weights
```

## Network Architecture

### QNetwork (Function Approximator)

3-layer fully connected neural network (MLP) implemented using **Flax NNX**:

```
Input (4)  →  FC (64)  →  ReLU  →  FC (64)  →  ReLU  →  FC (2)  →  Output
State vector   Hidden 1             Hidden 2            Q-values (left, right)
```

**Implementation Code:**
```python
class QNetwork(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 64, rngs=rngs)
        self.fc3 = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values (Logits)
```

**Input Example:**
```python
state = [0.02, 0.01, -0.03, 0.04]  # [position, velocity, angle, angular_velocity]
q_values = network(state)  # Output: [Q(s, left), Q(s, right)] = [1.23, 2.45]
action = argmax(q_values)  # Choose action with highest Q-value → right (1)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STATE_DIM` | 4 | State space dimension |
| `ACTION_DIM` | 2 | Action space dimension |
| `BUFFER_SIZE` | 10,000 | Replay Buffer capacity |
| `BATCH_SIZE` | 64 | Training batch size |
| `GAMMA` | 0.99 | Discount factor |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `NUM_EPISODES` | 500 | Number of training episodes |
| `TARGET_UPDATE_FREQ` | 100 | Target network update frequency (steps) |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.01 | Final exploration rate |
| `EPSILON_DECAY` | 0.995 | Epsilon decay rate (per episode) |

### Epsilon Decay Strategy

Uses **Exponential Decay**:

```python
epsilon = max(EPSILON_END, epsilon × EPSILON_DECAY)
```

This differs from Q-Learning's **linear decay**, enabling faster transition from "exploration" to "exploitation."

## DQN Agent Core Workflow

### 1. Initialization

```python
agent = DQNAgent(STATE_DIM, ACTION_DIM, rng_key=rng_key)
```

Creates:
- Online Network (trainable)
- Target Network (weights frozen)
- Replay Buffer (experience pool)
- Optimizer (Adam)

### 2. Action Selection (Epsilon-Greedy)

```python
def select_action(self, state, rng_key):
    if random() <= epsilon:
        return random_action()  # Explore
    else:
        q_values = self.online_network(state)
        return argmax(q_values)  # Exploit
```

### 3. Training Step

```python
def train_step(self):
    # 1. Sample from Replay Buffer
    states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

    # 2. Compute TD Target (using Target Network)
    q_next_target = self.target_network(next_states)
    td_target = rewards + GAMMA * max(q_next_target) * (1 - dones)

    # 3. Define Loss Function
    def loss_fn(model):
        q_current = model(states)
        q_current_action = q_current[actions]
        return mean((q_current_action - td_target)²)

    # 4. Compute Gradients and Update Online Network
    _, grads = nnx.value_and_grad(loss_fn)(self.online_network)
    self.optimizer.update(grads)
```

### 4. Main Training Loop

```python
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    while not done:
        # (1) Select action
        action = agent.select_action(state, rng_key)

        # (2) Execute action
        next_state, reward, done, _, _ = env.step(action)

        # (3) Store experience in Replay Buffer
        agent.buffer.add(state, action, reward, next_state, done)

        # (4) Train Online Network
        agent.train_step()

        # (5) Periodically update Target Network
        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        state = next_state
        episode_reward += reward

    # (6) Epsilon decay
    agent.update_epsilon()
```

## Expected Output

### Training Process

The program outputs training progress every 50 episodes:

```
Starting DQN Agent training...
Episode 50, Epsilon: 0.779, Avg Reward (last 50): 22.34
Episode 100, Epsilon: 0.606, Avg Reward (last 50): 45.12
Episode 150, Epsilon: 0.472, Avg Reward (last 50): 98.56
...Synchronizing Target Network weights...
Episode 200, Epsilon: 0.368, Avg Reward (last 50): 165.78
Episode 250, Epsilon: 0.286, Avg Reward (last 50): 234.12
...Synchronizing Target Network weights...
Episode 300, Epsilon: 0.223, Avg Reward (last 50): 312.45
Episode 350, Epsilon: 0.174, Avg Reward (last 50): 421.67
Episode 400, Epsilon: 0.135, Avg Reward (last 50): 487.23  ← Success!
Training completed!
```

**Interpretation:**
- **Epsilon** continuously decreases: transition from exploration to exploitation
- **Avg Reward** gradually increases: agent gets smarter
- **Target network sync**: synchronization messages appear periodically
- **Success criteria**: average reward ≥ 475

## Flax NNX Key APIs

This implementation uses **Flax NNX** (next-generation API) instead of legacy `flax.linen`:

### 1. Model Definition

```python
class QNetwork(nnx.Module):
    def __init__(self, ..., *, rngs: nnx.Rngs):  # Must receive rngs
        self.fc1 = nnx.Linear(...)
```

### 2. Optimizer Binding

```python
self.optimizer = nnx.Optimizer(self.online_network, optax.adam(LEARNING_RATE))
```

### 3. Weight Extraction and Update (Target Network Sync)

```python
# Extract Online Network weights
online_state = nnx.state(self.online_network)

# Update Target Network weights
nnx.update(self.target_network, online_state)
```

### 4. Gradient Computation and Update

```python
# Compute loss and gradients
_, grads = nnx.value_and_grad(loss_fn)(self.online_network)

# Update parameters using optimizer
self.optimizer.update(grads)
```

## Q-Learning vs DQN Comparison

| Feature | Q-Learning (Tabular) | DQN (Deep Learning) |
|---------|---------------------|---------------------|
| **Q-value Storage** | Q-Table (NumPy array) | Neural Network |
| **State Space** | Small discrete states (16) | High-dim continuous states (infinite) |
| **Memory Requirements** | O(states × actions) | O(network parameters) |
| **Generalization** | ❌ None (each state independent) | ✅ Yes (similar states share knowledge) |
| **Experience Replay** | ❌ Not used | ✅ Replay Buffer |
| **Target Network** | ❌ Not needed | ✅ Target Network |
| **Update Method** | Direct Q(s,a) update | Gradient descent |
| **Epsilon Decay** | Linear decay | Exponential decay |

## Advanced Techniques & Extensions

This implementation uses basic DQN. Here are improved versions:

1. **Double DQN** (DDQN)
   - Solves Q-value overestimation problem
   - Uses Online Network to select action, Target Network to evaluate value

2. **Dueling DQN**
   - Network splits into two branches: V(s) and A(s,a)
   - Better value estimation

3. **Prioritized Experience Replay** (PER)
   - Prioritizes "important" experiences for replay
   - Accelerates learning

4. **Rainbow DQN**
   - Combines all the above techniques for strongest version

## From DQN to Policy Gradients

**DQN Limitations:**
- ❌ Cannot handle **continuous action spaces** (e.g., steering angle -180° ~ 180°)
- ❌ Only "indirectly" learns policy (through Q-values)

**Next Step: Policy Gradient Methods**
- See `3_Pendulum/` for PPO implementation
- Directly learn policy π(a|s)
- Can handle continuous actions

## References

- Mnih et al. (2015). "Human-level control through deep reinforcement learning" (Nature)
- Mnih et al. (2013). "Playing Atari with Deep Reinforcement Learning" (NIPS Workshop)
- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 9-11)
- [Gymnasium CartPole-v1 Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
