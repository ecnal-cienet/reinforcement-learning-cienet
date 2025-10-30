# Proximal Policy Optimization (PPO) Implementation - Pendulum-v1

> **中文版本**: [README_ch.md](README_ch.md)

## Overview

This is a complete implementation of the **Proximal Policy Optimization (PPO)** algorithm solving OpenAI Gymnasium's **Pendulum-v1** environment. PPO is the core algorithm in modern reinforcement learning, widely applied in robot control, game AI, and **large language model alignment training (RLHF)**.

**Core Breakthrough:** PPO combines **Actor-Critic architecture**, **continuous action space handling**, and **stable policy update mechanisms**, making it one of the most popular RL algorithms in industry today.

**Relationship to Previous Projects:**
- `1_Q_Learning/`: Learn Q-values (value-based) - Tabular
- `2_Cart_Pole_DQN/`: Learn Q-values (value-based) - Deep learning + discrete actions
- `3_Pendulum/` **(This project)**: Learn policy (policy-based) - Deep learning + **continuous actions**

## Environment Description

### Pendulum-v1 (Inverted Pendulum)

Pendulum is a classic continuous control problem: a pendulum starts from a random position, and the goal is to apply appropriate torque to keep the pendulum upright at the **top** position.

```
    ↑ Target position
    |
    |
    O ← Pivot
   /
  /  ← Pendulum
 ●

Goal: Apply torque to rotate pendulum to top and keep it stable
```

### Environment Parameters

- **State Space**: Continuous 3-dimensional vector
  - `cos(θ)`: Cosine of pendulum angle (range: -1 ~ 1)
  - `sin(θ)`: Sine of pendulum angle (range: -1 ~ 1)
  - `θ̇`: Angular velocity of pendulum (range: -8 ~ 8 rad/s)

  > **Why use cos/sin instead of angle?** Because angles have periodicity (0° = 360°), using cos/sin makes the state space smoother.

- **Action Space**: **Continuous** 1-dimensional vector
  - `torque`: Applied torque (range: **-2 ~ 2**)
  - ⚠️ **Key Difference**: This is **continuous action**, unlike CartPole's discrete "left/right" choices

- **Reward Function**:
  ```
  reward = -(θ² + 0.1 × θ̇² + 0.001 × torque²)
  ```
  - Penalizes deviation from upright position (θ²)
  - Penalizes excessive angular velocity (θ̇²)
  - Penalizes excessive torque (torque², encourages energy efficiency)
  - **Range**: approximately -16.3 (worst) ~ 0 (perfect)

- **Termination Conditions**:
  - No early termination
  - Each episode has fixed 200 steps

- **Success Criteria**:
  - Average reward > -200 indicates basic success
  - Average reward > -150 indicates good control

## How to Run

### Prerequisites

Ensure you've activated the virtual environment and installed dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** This project requires `tensorflow-probability` to handle continuous action probability distributions.

### Execute the Program

```bash
python 3_Pendulum/pendulum.py
```

Or from within the `3_Pendulum` directory:

```bash
cd 3_Pendulum
python pendulum.py
```

## Algorithm Core

### Evolution from DQN to PPO

#### Two Major Limitations of DQN

1. **Cannot Handle Continuous Action Spaces**
   - DQN relies on `argmax` operation: `action = argmax Q(s, a)`
   - In Pendulum, actions are **any real number between -2.0 and 2.0**
   - You cannot take `max` over "infinitely many" actions

2. **Can Only Learn Policy Indirectly**
   - DQN learns "Q-values", and policy π is "derived" from Q-values
   - What we really want is the "**policy itself**"

#### PPO's Solution

**Core Idea:** Directly learn a "**Policy Network**", denoted π<sub>θ</sub>(a|s).

- **Input**: State s
- **Output**: **Probability distribution** of actions (not a single action)
  - **Discrete actions (CartPole)**: `[P(left), P(right)]` = `[0.3, 0.7]`
  - **Continuous actions (Pendulum)**: A **Normal distribution** `N(μ, σ²)`
    - `μ` (mean): Most likely action
    - `σ` (standard deviation): Degree of exploration

**Example:**
```python
state = [cos(θ), sin(θ), θ̇] = [0.8, 0.6, 1.2]
distribution = actor(state)  # → N(μ=1.5, σ=0.3)
action = distribution.sample()  # Sample from distribution → might get 1.7
```

### Actor-Critic Architecture

PPO uses **two** neural networks working together:

#### 1. Actor - Policy Network π<sub>θ</sub>

**Job:** Decision maker (outputs actions)

**Network Structure:** Dual-head MLP
```
Input (3)  →  FC(64) → ReLU → FC(64) → ReLU → ┬→ FC_mu(1)    → tanh × 2 → μ
                                                 └→ FC_sigma(1) → softplus → σ
```

**Key Design:**
1. **μ head (mean)**:
   - Use `tanh` to compress output to [-1, 1]
   - Multiply by 2 → range becomes [-2, 2] (matches environment requirements)

2. **σ head (standard deviation)**:
   - Use `softplus` to ensure σ > 0 (standard deviation must be positive)
   - Add 1e-5 to avoid numerical instability

**Output:** `tfp.distributions.Normal(loc=μ, scale=σ)`

**Code:**
```python
class Actor(nnx.Module):
    def __call__(self, x: jax.Array) -> tfd.Normal:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))

        mu = jnp.tanh(self.fc_mu(x)) * 2.0      # Mean [-2, 2]
        sigma = nnx.softplus(self.fc_sigma(x)) + 1e-5  # Std > 0

        return tfd.Normal(loc=mu, scale=sigma)   # Return probability distribution
```

#### 2. Critic - Value Network V<sub>φ</sub>

**Job:** Evaluator (provides baseline)

**Network Structure:** Standard MLP
```
Input (3)  →  FC(64) → ReLU → FC(64) → ReLU → FC_out(1) → V(s)
```

**Output:** A single number representing "in state s, expected total reward I can get"

**Code:**
```python
class Critic(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc_out(x)  # Output V(s)
```

### PPO's Three Core Techniques

#### Technique 1: Advantage (Advantage Function)

**Problem:** REINFORCE uses "absolute total score" as learning signal → too noisy

**Solution:** Use "relative score"

```
Advantage(s, a) = Actual score received - Critic's expected score
A(s, a) = Q(s, a) - V(s)
```

**Signal Interpretation:**
- `A > 0`: Performance **better than expected** → increase this action's probability ✅
- `A < 0`: Performance **worse than expected** → decrease this action's probability ❌
- `A ≈ 0`: Performance **as expected** → no change

#### Technique 2: GAE (Generalized Advantage Estimation)

**Problem:** How to accurately compute Advantage?

**Solution:** Use GAE, a "smooth" Advantage calculation method

**GAE Formula (iterate backwards recursively):**
```python
for t in reversed(range(N)):
    # 1. Compute TD error
    delta_t = reward_t + γ × V(s_{t+1}) - V(s_t)

    # 2. Compute GAE (recursive)
    A_t = delta_t + γ × λ × A_{t+1}

    # 3. Compute Return (Critic's learning target)
    Return_t = A_t + V(s_t)
```

**Hyperparameters:**
- `γ` (GAMMA = 0.99): Discount factor (importance of future rewards)
- `λ` (GAE_LAMBDA = 0.95): GAE smoothing parameter
  - `λ = 0`: Only look one step ahead (low variance, high bias)
  - `λ = 1`: Look to the end (high variance, low bias)
  - `λ = 0.95`: Compromise ⭐

**Final Optimization: Advantage Normalization**
```python
advantages = (advantages - mean) / (std + 1e-8)
```
Makes Advantage have mean 0, standard deviation 1 → more stable training

#### Technique 3: PPO-Clip (Limit Update Steps)

**Problem:** Actor-Critic training is unstable, may "step too far" causing policy collapse

**Solution:** PPO-Clip adds "safety lock"

**Core Concept: Policy Ratio**
```
Ratio = π_new(a|s) / π_old(a|s)
```
- `Ratio ≈ 1`: New and old policies similar (safe)
- `Ratio >> 1` or `Ratio << 1`: New and old policies too different (dangerous)

**PPO-Clip Loss Function:**
```python
# Compute two types of Loss
loss_unclipped = Advantage × Ratio
loss_clipped = Advantage × clip(Ratio, 1-ε, 1+ε)

# Take smaller value (pessimistic principle)
loss = -mean(minimum(loss_unclipped, loss_clipped))
```

**CLIP_EPSILON = 0.2 Meaning:**
- Ratio is limited to [0.8, 1.2] range
- Even if Advantage is large, policy **cannot** update by more than 20% at once
- Ensures stable training

**Visualization:**
```
Advantage > 0 (good action)
┌────────────────────────────┐
│  Allow increase prob, max 20%  │  ← Clip upper limit (1.2)
├────────────────────────────┤
│  Normal update range [0.8, 1.2]    │
├────────────────────────────┤
│  Allow decrease prob, max 20%  │  ← Clip lower limit (0.8)
└────────────────────────────┘

Advantage < 0 (bad action) - reverse
```

## PPO Training Workflow

### On-Policy vs Off-Policy

| Feature | Off-Policy (DQN) | On-Policy (PPO) |
|---------|-----------------|-----------------|
| **Data Source** | Any old policy | Must be **current** policy |
| **Experience Replay** | ✅ Replay Buffer (reusable) | ❌ Rollout Buffer (use once, discard) |
| **Training Stability** | Harder (needs Target Network) | Easier (smoother policy updates) |
| **Sample Efficiency** | High (reuse data multiple times) | Low (use data once) |

**Why is PPO On-Policy?**
- PPO's Loss calculation needs "old policy's log probability"
- If data comes from "too old" policy, Ratio will be distorted
- Therefore, PPO must **immediately learn after collecting data, then discard**

### RolloutBuffer (On-Policy Storage)

```python
class RolloutBuffer:
    def add(self, state, action, reward, log_prob, value, done):
        # Store one step's experience

    def calculate_advantages_and_returns(self, last_value, gamma, gae_lambda):
        # Compute GAE and Returns (iterate backwards)

    def get_data_for_learning(self):
        # Convert to JAX arrays for training

    def clear(self):
        # After learning, clear all data
```

### PPO Four-Phase Lifecycle

```python
while total_steps < MAX_STEPS:
    # ========== Phase 1: Collect (Rollout) ==========
    for _ in range(ROLLOUT_STEPS):  # e.g., 2048 steps
        # 1. Select action
        action, value, log_prob = agent.select_action(state)

        # 2. Interact with environment
        next_state, reward, done, _, _ = env.step(action)

        # 3. Store in Buffer
        buffer.add(state, action, reward, log_prob, value, done)

        state = next_state

    # ========== Phase 2: Compute Learning Targets (GAE) ==========
    # Get "last step" V value
    last_value = critic(state)

    # Compute Advantages and Returns for all steps
    buffer.calculate_advantages_and_returns(last_value, GAMMA, GAE_LAMBDA)

    # ========== Phase 3: Learn ==========
    # Get all data
    states, actions, log_probs_old, advantages, returns = buffer.get_data_for_learning()

    # Train K times (TRAIN_EPOCHS = 10)
    for epoch in range(TRAIN_EPOCHS):
        # Shuffle data
        indices = random.permutation(ROLLOUT_STEPS)

        # Train in batches (BATCH_SIZE = 64)
        for batch_indices in batches(indices, BATCH_SIZE):
            # Train Critic (minimize MSE)
            train_critic(batch_states, batch_returns)

            # Train Actor (PPO-Clip Loss)
            train_actor(batch_states, batch_actions,
                       batch_log_probs_old, batch_advantages)

    # ========== Phase 4: Discard ==========
    buffer.clear()  # Clear all "old policy" data
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STATE_DIM` | 3 | State space dimension (cos θ, sin θ, θ̇) |
| `ACTION_DIM` | 1 | Action space dimension (torque) |
| `NUM_TOTAL_TIMESTEPS` | 100,000 | Total training steps |
| `ROLLOUT_STEPS` | 2,048 | Steps collected each time (N) |
| `TRAIN_EPOCHS` | 10 | Training epochs per batch (K) |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `GAMMA` | 0.99 | Discount factor (γ) |
| `GAE_LAMBDA` | 0.95 | GAE smoothing parameter (λ) |
| `CLIP_EPSILON` | 0.2 | PPO clipping parameter (ε) |
| `ACTOR_LR` | 3e-4 | Actor learning rate |
| `CRITIC_LR` | 1e-3 | Critic learning rate |

**Learning Rate Choice:**
- Critic's learning rate (1e-3) is higher than Actor's (3e-4)
- Reason: Critic needs to quickly learn to evaluate states, providing accurate baseline for Actor

## Expected Output

### Training Process

```
Starting PPO training...

--- Collecting 2048 steps of data ---
...Calculating GAE (Advantages) and Returns...
...Starting 10 Epochs of learning...
Current total steps: 2048/100000

--- Collecting 2048 steps of data ---
...Calculating GAE (Advantages) and Returns...
...Starting 10 Epochs of learning...
Current total steps: 4096/100000

...

Current total steps: 100000/100000
--- Training completed! ---
```

**Interpretation:**
- Collect 2048 steps each time
- Compute GAE
- Train for 10 Epochs (each Epoch uses all 2048 data points, split into multiple batches)
- Discard data and start next round

### Evaluating Training Results

After training, manually evaluate the agent's performance:

```python
# Add at end of main() function
env = gym.make("Pendulum-v1", render_mode="human")
state, _ = env.reset()

for _ in range(1000):
    action, _, _ = agent.select_action(state)
    state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()
```

**Success Indicators:**
- Pendulum can quickly rotate to top
- Stay stable at top (small oscillations)
- Average reward > -200

## Core Code Analysis

### 1. Actor's select_action (JAX ↔ NumPy Bridge)

```python
def select_action(self, state: np.ndarray):
    # NumPy → JAX (add batch dimension)
    state_jnp = jnp.asarray(state[np.newaxis, :], dtype=jnp.float32)

    # Call Actor → get probability distribution
    action_dist = self.actor(state_jnp)  # N(μ, σ)

    # Call Critic → get baseline
    value = self.critic(state_jnp)  # V(s)
    value = jax.lax.stop_gradient(value)  # Stop gradient backprop

    # Sample action from distribution
    action = action_dist.sample(seed=rng_key)

    # Compute log probability (PPO must have)
    log_prob = action_dist.log_prob(action)

    # JAX → NumPy (remove batch dimension)
    return action.flatten(), value.flatten(), log_prob.flatten()
```

**Why do we need log_prob?**
- PPO needs to compute `Ratio = exp(log_prob_new - log_prob_old)`
- Must record log_prob_old at "sampling time"

### 2. Critic Training (MSE Loss)

```python
def critic_loss_fn(critic_model: Critic):
    values_pred = critic_model(batch_states)  # Predicted V(s)
    loss = jnp.mean((batch_returns - values_pred.flatten()) ** 2)
    return loss

# Compute gradients and update
_, critic_grads = nnx.value_and_grad(critic_loss_fn)(self.critic)
self.critic_optimizer.update(critic_grads)
```

**Goal:** Make V(s) as close to "actual total score" (Returns) as possible

### 3. Actor Training (PPO-Clip Loss)

```python
def actor_loss_fn(actor_model: Actor):
    # 1. Get new log probabilities
    action_dist_new = actor_model(batch_states)
    log_probs_new = action_dist_new.log_prob(batch_actions)

    # 2. Compute Ratio
    ratio = jnp.exp(log_probs_new - batch_log_probs_old)

    # 3. Compute Unclipped Loss
    loss_unclipped = batch_advantages * ratio

    # 4. Compute Clipped Loss
    ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    loss_clipped = batch_advantages * ratio_clipped

    # 5. Take minimum (pessimistic principle)
    loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))
    return loss

# Compute gradients and update
_, actor_grads = nnx.value_and_grad(actor_loss_fn)(self.actor)
self.actor_optimizer.update(actor_grads)
```

**Key:** Add negative sign `-` because Adam can only "minimize", but we want to "maximize" Advantage

## Q-Learning → DQN → PPO Evolution Summary

| Feature | Q-Learning | DQN | PPO |
|---------|-----------|-----|-----|
| **Learning Target** | Q-value (value) | Q-value (value) | Policy |
| **Function Approx** | ❌ Q-Table | ✅ Neural Network | ✅ Neural Network (A+C) |
| **Action Space** | Discrete | Discrete | **Continuous + Discrete** |
| **Policy Type** | Off-Policy | Off-Policy | **On-Policy** |
| **Experience Replay** | ❌ | ✅ Replay Buffer | ❌ (Rollout Buffer) |
| **Stabilization** | ❌ | Target Network | **PPO-Clip + GAE** |
| **Advantage Function** | ❌ | ❌ | ✅ |
| **Application** | Small state space | Large state + discrete | **Any scenario** (most general) |

## PPO's Real-World Applications

PPO is currently the most popular RL algorithm in industry, with applications including:

1. **Robot Control**
   - Robotic arm grasping
   - Quadruped robot walking
   - Drone flight

2. **Game AI**
   - OpenAI Five (Dota 2)
   - AlphaStar (StarCraft II)
   - Various continuous control games

3. **Large Language Model Alignment (RLHF)**
   - ChatGPT training
   - Claude training
   - **GRPO** (Group Relative Policy Optimization) is a PPO variant

4. **Autonomous Driving**
   - Path planning
   - Speed control

## Advanced Topics

### PPO Variants

1. **PPO-Penalty**
   - Use KL divergence penalty instead of Clip
   - `Loss = Advantage - β × KL(π_new || π_old)`

2. **GRPO** (for LLMs)
   - Group Relative Policy Optimization
   - PPO variant designed specifically for large language models

### Further Optimizations

1. **Vectorized Environments**
   - Run multiple environment copies simultaneously
   - Accelerate data collection

2. **Normalization**
   - State normalization
   - Reward normalization

3. **Learning Rate Scheduling**
   - Learning rate decay
   - Improve training stability in later stages

## Flax NNX Key API Summary

### 1. Multiple Network Management

```python
# Create two independent networks
actor_key, critic_key = jax.random.split(rng_key)
self.actor = Actor(..., rngs=nnx.Rngs(actor_key))
self.critic = Critic(..., rngs=nnx.Rngs(critic_key))
```

### 2. Multiple Optimizer Management

```python
# Each network has its own optimizer
self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(3e-4))
self.critic_optimizer = nnx.Optimizer(self.critic, optax.adam(1e-3))
```

### 3. Gradient Blocking

```python
# Block Critic's gradient when select_action
value = self.critic(state_jnp)
value = jax.lax.stop_gradient(value)  # Don't train Critic
```

### 4. RNG Stream Management

```python
# Create RNG stream
self.rng_stream = nnx.Rngs(jax.random.PRNGKey(42))

# Get new key when randomness needed
rng_key = self.rng_stream.sampler()
action = action_dist.sample(seed=rng_key)
```

## References

- Schulman et al. (2017). "Proximal Policy Optimization Algorithms" ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
- Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation" ([arXiv:1506.02438](https://arxiv.org/abs/1506.02438))
- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 13: Policy Gradient Methods)
- [Gymnasium Pendulum-v1 Documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
