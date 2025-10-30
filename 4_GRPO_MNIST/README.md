# Group Relative Policy Optimization (GRPO) Implementation - MNIST Classification

> **中文版本**: [README_ch.md](README_ch.md)

## Overview

This is an implementation of the **Group Relative Policy Optimization (GRPO)** algorithm applied to **MNIST handwritten digit classification**. GRPO is a variant of PPO designed specifically for large language model (LLM) training, with the core innovation of **removing the Critic network** to save memory.

**Core Breakthrough:** GRPO uses "group average performance" as the baseline, replacing PPO's separately trained Critic network, thereby significantly reducing memory requirements.

**Relationship to Previous Projects:**
- `1_Q_Learning/`: Value-based - Tabular
- `2_Cart_Pole_DQN/`: Value-based - Deep learning
- `3_Pendulum/`: Policy-based - Actor-Critic (PPO)
- `4_GRPO_MNIST/` **(This project)**: Policy-based - **Critic-less** (GRPO)

## Why Do We Need GRPO?

### PPO's Memory Bottleneck in Large Model Training

When training large language models like ChatGPT (e.g., 8B parameters), PPO faces severe memory problems:

**PPO requires:**
1. **Actor Network** (8 billion parameters)
   - Weights: 16 GB
   - Gradients: 16 GB
   - Adam state: 32 GB
   - **Subtotal**: 64 GB

2. **Critic Network** (8 billion parameters)
   - Weights: 16 GB
   - Gradients: 16 GB
   - Adam state: 32 GB
   - **Subtotal**: 64 GB

**Total requirement**: 128 GB (training state only)

This is infeasible on a single chip (typically 32-80 GB HBM), and expensive even with distributed training.

### GRPO's Solution

**Core Idea:** Completely remove the Critic network, use "**Group Relative Baseline**"

**GRPO baseline calculation:**
```python
# PPO: Need to train a Critic network to predict V(s)
baseline = Critic(state)  # ← An 8 billion parameter network!

# GRPO: Dynamically compute group average
baseline = mean(group_rewards)  # ← Just a mean() operation!
```

**Advantages:**
- ✅ **Memory savings**: Saves all memory required for the Critic network (64 GB)
- ✅ **Training simplification**: No need to train and maintain Critic
- ✅ **No Critic training instability**: Avoids the problem of Critic misleading Actor

**Costs:**
- ❌ **Higher collection cost**: Need to generate `G` responses for each state to compute group average
- ❌ **Lower sample efficiency**: Same state requires multiple inferences

**Conclusion:** GRPO is an algorithm that **sacrifices inference cost (data collection) in exchange for memory efficiency (training)**, particularly suitable for LLM scenarios where single inference cost is relatively low but model training memory requirements are extremely high.

## Environment Description

### MNIST as an RL Problem

We reframe the classic supervised learning problem (MNIST classification) as a reinforcement learning problem:

**Traditional Supervised Learning:**
```
Input image → Model prediction → Compute Loss (Cross-Entropy) → Update weights
```

**RL Framework (GRPO):**
```
State (image) → Actor samples action (predicted digit) → Receive Reward (correct/incorrect) → Compute Advantage → Update Actor
```

### RL Environment Parameters

- **State Space**: Continuous 784-dimensional vector
  - 28×28 grayscale image flattened to (784,)
  - Value range: 0.0 ~ 1.0 (after normalization)

- **Action Space**: Discrete 10 actions
  - Actions 0-9 represent predicted digits
  - Uses `tfp.distributions.Categorical` to represent action probability distribution

- **Reward Function**:
  ```python
  reward = 1.0  # Correct prediction
  reward = 0.0  # Incorrect prediction
  ```
  - This is a "sparse reward" problem
  - Only completely correct answers receive reward

- **Episode Structure**:
  - **One-Step Episode**: Each image is an independent episode
  - Episode ends immediately after guessing the digit
  - No need to consider temporal dependencies

- **Success Criteria**:
  - Training set accuracy > 95% (good)
  - Training set accuracy > 97% (excellent)

## How to Run

### Prerequisites

Ensure you've activated the virtual environment and installed dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** This project requires the following additional dependencies:
- `tensorflow-datasets`: Load MNIST dataset
- `tensorflow-probability[jax]`: Provide `Categorical` distribution

### Execute the Program

```bash
python 4_GRPO_MNIST/grpo_mnist.py
```

Or from within the `4_GRPO_MNIST` directory:

```bash
cd 4_GRPO_MNIST
python grpo_mnist.py
```

## Algorithm Core

### GRPO vs PPO Comparison

| Feature | PPO | GRPO |
|---------|-----|------|
| **Network Structure** | Actor + Critic | **Actor only** |
| **Baseline** | `V(s)` (Critic network prediction) | `mean(group_rewards)` (dynamically computed) |
| **Advantage Calculation** | `A = R - V(s)` | `A = R - mean(R_group)` |
| **Memory Requirements** | High (two networks) | **Low (one network)** |
| **Collection Cost** | 1× (generate once per s) | **G× (generate G times per s)** |
| **Application Scenario** | General RL | **Large model training (LLM)** |

### GRPO's "Group" Concept

**Core Question:** Without a Critic, how do we judge if a Reward is "good" or "bad"?

**GRPO's Answer:** Use "group relative comparison"

**Example:**
```
Suppose we have a batch (G=1024) of images:
- Image 1: Agent guesses "7", correct → R₁ = 1.0
- Image 2: Agent guesses "3", incorrect → R₂ = 0.0
- Image 3: Agent guesses "5", correct → R₃ = 1.0
- ...
- Image 1024: Agent guesses "2", incorrect → R₁₀₂₄ = 0.0

Group average: baseline = mean([R₁, R₂, ..., R₁₀₂₄]) = 0.78

Relative advantage:
- Image 1: A₁ = 1.0 - 0.78 = +0.22 (Good! 22% better than average)
- Image 2: A₂ = 0.0 - 0.78 = -0.78 (Bad! 78% worse than average)
- Image 3: A₃ = 1.0 - 0.78 = +0.22 (Good!)
```

**Key Insight:**
- Even if "correct" (R=1.0), if group average is already high (e.g., 0.95), Advantage is only +0.05 (small update magnitude)
- Even if "incorrect" (R=0.0), if group average is also low (e.g., 0.20), Advantage is only -0.20 (lighter penalty)
- This "relative comparison" allows the Actor to learn "relative performance at current ability level"

### GRPO's Four-Phase Training Workflow

```python
for epoch in range(NUM_EPOCHS):
    for batch in mnist_dataset:  # Each batch is a "group"

        # ========== Phase 1: Collect (Rollout) ==========
        # Sample actions for each image in the group
        actions, log_probs_old = actor.select_actions_and_log_probs(images)

        # ========== Phase 2: Compute Relative Advantage (GRPO Core) ==========
        # (A) Compute rewards
        rewards = (actions == labels).astype(float)  # Correct or incorrect

        # (B) Compute group baseline (replaces Critic)
        baseline = mean(rewards)

        # (C) Compute relative advantage
        advantages = rewards - baseline

        # (D) Normalize (same as PPO)
        advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)

        # ========== Phase 3: Learn (PPO-Clip Loss) ==========
        # Use the same Clip Loss as PPO to train Actor
        train_actor(images, actions, log_probs_old, advantages)
```

**Key Differences:**
1. ❌ **No RolloutBuffer**: Because it's a one-step problem, no need to store multi-step experiences
2. ❌ **No GAE calculation**: Because there's no temporal dependency, directly use `R - baseline`
3. ❌ **No Critic training**: GRPO's core feature
4. ✅ **Retains PPO-Clip**: Ensures stable policy updates

## Network Architecture

### Actor (The Only Neural Network)

3-layer fully connected neural network (MLP) implemented using **Flax NNX**:

```
Input (784)  →  FC(128)  →  ReLU  →  FC(128)  →  ReLU  →  FC(10)  →  Categorical
Image vector    Hidden 1              Hidden 2             Logits     Prob dist
```

**Output:** `tfp.distributions.Categorical(logits)`
- 10 logits (raw scores) are automatically converted to probability distribution
- Example: `[0.1, 0.05, 0.3, 0.02, ...]` (probabilities for 10 digits)

**Code:**
```python
class Actor(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 128, rngs=rngs)
        self.fc_out = nnx.Linear(128, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> tfd.Categorical:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        logits = self.fc_out(x)
        return tfd.Categorical(logits=logits)  # Return discrete probability distribution
```

**Example:**
```python
image = [0.1, 0.2, ..., 0.05]  # 784-dimensional vector
dist = actor(image)             # Categorical distribution
action = dist.sample()          # Sample → might get 7
log_prob = dist.log_prob(action)  # log P(action=7|state)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STATE_DIM` | 784 | State space dimension (28×28) |
| `ACTION_DIM` | 10 | Action space dimension (0-9) |
| `NUM_EPOCHS` | 10 | Number of training epochs |
| `BATCH_SIZE` | 1,024 | **Group size (G)** |
| `LEARNING_RATE` | 1e-4 | Actor learning rate |
| `CLIP_EPSILON` | 0.2 | PPO clipping parameter (ε) |

**BATCH_SIZE's Dual Meaning:**
1. **Traditional meaning**: Amount of data processed per batch (Mini-batch)
2. **GRPO meaning**: **Group size (G)**, used to compute relative baseline

**Why is larger BATCH_SIZE (1024) better?**
- Larger group → more stable `mean(rewards)`
- More accurate baseline → cleaner Advantage signal
- But also increases memory and computational cost

## Core Code Analysis

### 1. Action Selection and Log Probability Calculation

```python
def select_actions_and_log_probs(self, batch_states: jax.Array):
    # 1. Get probability distribution
    action_dist = self.actor(batch_states)  # Categorical(logits=[...])

    # 2. Sample actions (sample one digit for each of G=1024 images)
    rng_key = self.rng_stream.sampler()
    actions = action_dist.sample(seed=rng_key)

    # 3. Compute "old" log probabilities (PPO must have)
    log_probs = action_dist.log_prob(actions)

    return actions, log_probs
```

**Why do we need log_prob?**
- PPO-Clip needs to compute `ratio = π_new / π_old`
- Implementation uses `ratio = exp(log_prob_new - log_prob_old)` to avoid numerical instability

### 2. GRPO Relative Advantage Calculation

```python
# (A) Compute rewards (correct or incorrect)
@jax.jit
def calculate_rewards(actions, labels):
    return jnp.where(actions == labels, 1.0, 0.0)

batch_rewards = calculate_rewards(batch_actions, batch_labels)

# (B) Compute group baseline (GRPO core!)
baseline = jnp.mean(batch_rewards)  # ← Replaces Critic(state)

# (C) Compute relative advantage
batch_advantages = batch_rewards - baseline

# (D) Normalize (same as PPO)
adv_mean = jnp.mean(batch_advantages)
adv_std = jnp.std(batch_advantages) + 1e-8
batch_advantages = (batch_advantages - adv_mean) / adv_std
```

**Key Insight:**
```python
# PPO needs to additionally train Critic
baseline = critic(state)  # 8 billion parameters

# GRPO only needs a mean operation
baseline = mean(rewards)  # Zero parameters!
```

### 3. Actor Training (PPO-Clip Loss)

```python
def actor_loss_fn(actor_model: Actor):
    # (1) Get new probability distribution
    action_dist_new = actor_model(batch_states)
    log_probs_new = action_dist_new.log_prob(batch_actions)

    # (2) Compute policy ratio
    ratio = jnp.exp(log_probs_new - batch_log_probs_old)

    # (3) Compute unclipped loss
    loss_unclipped = batch_advantages * ratio

    # (4) Compute clipped loss (PPO core)
    ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    loss_clipped = batch_advantages * ratio_clipped

    # (5) Take smaller value (pessimistic principle)
    loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))
    return loss

# Compute gradients and update
_, actor_grads = nnx.value_and_grad(actor_loss_fn)(self.actor)
self.actor_optimizer.update(actor_grads)
```

**This is exactly the same as PPO's Actor training!**
- The only difference is how Advantage is calculated (group relative vs Critic prediction)

## Expected Output

### Training Process

```
Starting GRPO on MNIST training...
Epoch 1/10, Accuracy: 85.32%
Epoch 2/10, Accuracy: 90.18%
Epoch 3/10, Accuracy: 92.47%
Epoch 4/10, Accuracy: 94.12%
Epoch 5/10, Accuracy: 95.23%
Epoch 6/10, Accuracy: 96.08%
Epoch 7/10, Accuracy: 96.54%
Epoch 8/10, Accuracy: 96.89%
Epoch 9/10, Accuracy: 97.12%
Epoch 10/10, Accuracy: 97.34%
--- Training completed! ---
```

**Interpretation:**
- **Early rapid improvement** (Epoch 1-3): Agent learns basic digit recognition
- **Mid-stage steady growth** (Epoch 4-6): Policy continues to optimize
- **Late-stage convergence** (Epoch 7-10): Accuracy approaches limit
- **Success criteria**: Accuracy > 95% indicates GRPO successfully applied to supervised learning problem

### Performance Analysis

**Memory savings:**
- PPO: Needs Actor (128→128→10) + Critic (128→128→1)
- GRPO: Only needs Actor (128→128→10)
- **Saves about 50% of network parameters** (savings more significant in LLM scenarios)

**Computational cost:**
- PPO: Sample once per sample
- GRPO: Need G samples per batch to compute group baseline
- **Inference cost increases by approximately G times** (but in LLM scenarios, inference cost << training memory cost)

## Q-Learning → DQN → PPO → GRPO Evolution Summary

| Feature | Q-Learning | DQN | PPO | GRPO |
|---------|-----------|-----|-----|------|
| **Learning Target** | Q-value | Q-value | Policy | Policy |
| **Function Approx** | Q-Table | Neural Network | Neural Network (A+C) | **Neural Network (A)** |
| **Action Space** | Discrete | Discrete | Continuous+Discrete | Continuous+Discrete |
| **Baseline Method** | ❌ None | ❌ None | Critic network | **Group average** |
| **Number of Networks** | 0 | 2 (Online+Target) | 2 (Actor+Critic) | **1 (Actor)** |
| **Memory Requirements** | Low | Medium | High | **Low** |
| **Sample Efficiency** | Low | High (Replay) | Low (On-Policy) | **Lowest (G×)** |
| **Application Scenario** | Small state space | Large state space | General RL | **Large model training** |

## GRPO in LLM Training Applications

### MNIST to LLM Analogy

| MNIST (This Project) | LLM (ChatGPT/Claude) |
|---------------------|---------------------|
| State: Image (784-dim) | State: Prompt + History (context) |
| Action: Predict digit (0-9) | Action: Generate next token |
| Reward: Correct/incorrect (0/1) | Reward: Human preference score (Reward Model) |
| Group size: BATCH_SIZE=1024 | Group size: Generate G responses per prompt |
| Baseline: `mean(group accuracy)` | Baseline: `mean(group reward scores)` |

### GRPO Workflow in LLMs

```python
# Assume training ChatGPT
for prompt in training_prompts:
    # ========== Phase 1: Collect (Multiple Samples) ==========
    # Generate G=8 different responses for the same prompt
    responses = []
    for _ in range(G=8):
        response = actor.generate(prompt)  # Sample from LLM
        responses.append(response)

    # ========== Phase 2: Compute Relative Advantage ==========
    # Score using Reward Model
    rewards = [reward_model(prompt, resp) for resp in responses]
    # rewards = [7.2, 8.5, 6.1, 9.0, 7.8, 6.5, 8.2, 7.5]

    # Compute group baseline
    baseline = mean(rewards)  # = 7.6

    # Compute relative advantage
    advantages = rewards - baseline
    # advantages = [-0.4, +0.9, -1.5, +1.4, +0.2, -1.1, +0.6, -0.1]

    # ========== Phase 3: Learn (PPO-Clip) ==========
    # Train LLM to increase probability of "good" responses, decrease "bad" responses
    train_actor(prompt, responses, advantages)
```

**Why is LLM training particularly suitable for GRPO?**
1. **Severe memory bottleneck**: 8 billion parameter Critic cost too high
2. **Relatively controllable inference cost**: Cost of generating 8 responses << cost of training 8 billion parameter Critic
3. **Reasonable group comparison**: Multiple responses to the same prompt naturally form "groups"

## Implementation Limitations & Extensions

### Simplifications in This Project

1. **One-Step Problem**
   - MNIST doesn't require temporal reasoning
   - Real LLMs are multi-step sequence generation problems (need to handle Advantage for each token)

2. **Sparse Rewards**
   - Only 0/1 rewards
   - Real LLMs use Reward Models providing continuous scores

3. **No Reference Model**
   - This implementation doesn't use KL divergence penalty
   - Real GRPO typically includes `BETA × KL(π_new || π_ref)` to further stabilize training

### Advanced Optimizations

1. **KL Divergence Constraint**
   ```python
   # Prevent policy from deviating too far from "base model"
   kl_penalty = beta × KL(actor || reference_model)
   loss = -advantages + kl_penalty
   ```

2. **Adaptive Baseline**
   ```python
   # Use moving average as baseline
   baseline = 0.9 × baseline + 0.1 × mean(rewards)
   ```

3. **Multi-Round Optimization**
   ```python
   # Like PPO, train multiple epochs on the same batch of data
   for epoch in range(K_EPOCHS):
       train_actor(...)
   ```

## References

- DeepSeek-R1 Technical Report (2025). "Group Relative Policy Optimization for RLHF"
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms" ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
- OpenAI. "Learning to Summarize from Human Feedback" (introduces RLHF workflow)
- Anthropic. "Training a Helpful and Harmless Assistant with RLHF" (Claude training methods)

## Summary

GRPO is a clever variant of PPO for large model training scenarios:

**Core Trade-off:**
- ✅ **Sacrifice**: Inference cost (need G samples)
- ✅ **Gain**: Memory efficiency (remove Critic network)

**Application Scenarios:**
- ✅ Large language model training (ChatGPT, Claude, Llama)
- ✅ Memory-constrained scenarios
- ❌ Small models (PPO more efficient)
- ❌ Scenarios where inference cost is extremely high

This MNIST project demonstrates GRPO's core concepts. Although it's a simplified one-step problem, the "group relative baseline" idea fully applies to LLM's multi-step sequence generation scenarios.
