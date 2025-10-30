# Q-Learning Implementation (Tabular Reinforcement Learning)

> **中文版本**: [README_ch.md](README_ch.md)

## Overview

This is a classic **Q-Learning** algorithm implementation applied to a simple **4x4 Grid World** environment. Q-Learning is a value-based, model-free reinforcement learning algorithm, particularly suitable for discrete environments with small state spaces.

## Environment Description

### Grid World

```
State Number Layout:
┌────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │  Start: State 0 (top-left)
├────┼────┼────┼────┤
│ 4  │ 5  │ 6  │ 7  │
├────┼────┼────┼────┤
│ 8  │ 9  │ 10 │ 11 │
├────┼────┼────┼────┤
│ 12 │ 13 │ 14 │ 15 │  Goal: State 15 (bottom-right)
└────┴────┴────┴────┘
```

### Environment Parameters

- **State Space**: 16 discrete states (0 to 15)
- **Action Space**: 4 actions
  - `0`: Up (↑)
  - `1`: Down (↓)
  - `2`: Left (←)
  - `3`: Right (→)
- **Start State**: State 0 (top-left corner)
- **Goal State**: State 15 (bottom-right corner)

### Reward Function

- **Reaching goal state (State 15)**: Reward `+100`
- **Each move**: Penalty `-1` (encourages finding shortest path)

## How to Run

### Prerequisites

Ensure you've activated the virtual environment and installed dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Execute the Program

```bash
python 1_Q_Learning/Q_Learning.py
```

Or from within the `1_Q_Learning` directory:

```bash
cd 1_Q_Learning
python Q_Learning.py
```

## Algorithm Core

### Q-Learning Update Formula

Q-Learning uses **Temporal-Difference (TD)** methods to update the Q-Table:

```
Q(S, A) ← Q(S, A) + α × [R + γ × max Q(S', a') - Q(S, A)]
                              └─────────────────┘
                                  TD Target
```

**Parameter Explanations:**
- **S**: Current state
- **A**: Action taken
- **R**: Immediate reward received
- **S'**: New state
- **α (alpha)**: Learning rate = `0.1` (controls trust in new information)
- **γ (gamma)**: Discount factor = `0.99` (controls importance of future rewards)

### Epsilon-Greedy Strategy

During training, the agent uses **Epsilon-Greedy** strategy to balance exploration and exploitation:

- **Exploration** (probability = ε): Randomly select an action
- **Exploitation** (probability = 1-ε): Select action with highest Q-value

**Epsilon Decay:**
- **Initial value (EPSILON)**: `1.0` (100% exploration)
- **Final value (MIN_EPSILON)**: `0.01` (1% exploration)
- **Decay episodes (DECAY_EPISODES)**: `10000` episodes

```python
EPSILON = MAX_EPSILON - (MAX_EPSILON - MIN_EPSILON) × (episode / DECAY_EPISODES)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ALPHA` | 0.1 | Learning rate |
| `GAMMA` | 0.99 | Discount factor |
| `NUM_EPISODES` | 20000 | Number of training episodes |
| `MAX_STEPS_PER_EPISODE` | 100 | Maximum steps per episode |
| `EPSILON` (initial) | 1.0 | Initial exploration rate |
| `MIN_EPSILON` | 0.01 | Minimum exploration rate |
| `DECAY_EPISODES` | 10000 | Episodes for epsilon decay |

## Expected Output

### Training Process

The program outputs the current epsilon value every 2000 episodes:

```
Episode: 2000, Epsilon: 0.8200
Episode: 4000, Epsilon: 0.6400
Episode: 6000, Epsilon: 0.4600
Episode: 8000, Epsilon: 0.2800
Episode: 10000, Epsilon: 0.1000
Episode: 12000, Epsilon: 0.0100
...
Training completed!
```

### Learned Policy

After training, the program outputs the learned optimal policy (the best action for each state):

```
--- Learned Optimal Policy ---
 ↓  →  →  ↓
 ↓  ↓  →  ↓
 →  →  →  ↓
 →  →  →  G
```

**Interpretation:**
- Each arrow represents the action with highest Q-value in that state
- `G` represents the goal state
- Ideally, the policy should guide the agent from any state to the goal

### Q-Table Example

The program also outputs the final Q-Table (16×4 array), where each row represents a state and each column represents the Q-value for that action.

## Implementation Details

### Core Functions

1. **`step(state, action)`**
   - Implements environment state transition logic
   - Input: Current state and action
   - Output: `(new state, reward, done)`

2. **Training Loop**
   - Outer loop: Iterate through all training episodes
   - Inner loop: Steps within each episode
   - At each step:
     1. Use Epsilon-Greedy to select action
     2. Execute action and receive feedback
     3. Update Q-Table
     4. Update state

### Data Structure

- **Q-Table**: `numpy.ndarray` (shape: 16×4)
  - Row index: State number (0-15)
  - Column index: Action number (0-3)
  - Value: Q(state, action)

## Key Learning Points

This implementation demonstrates core RL concepts:

1. ✅ **Agent-Environment Interaction Loop**: State → Action → Reward → New State
2. ✅ **Temporal-Difference Learning (TD Learning)**: Update values without waiting for episode end
3. ✅ **Exploration vs. Exploitation Trade-off**: Epsilon-Greedy strategy
4. ✅ **Value Function Approximation**: Use Q-Table to store value for each state-action pair
5. ✅ **Policy Improvement**: Extract optimal policy from learned Q-values

## Limitations & Extensions

### Limitations of Tabular Methods

- **Curse of Dimensionality**: Cannot handle slightly larger state spaces
- **No Generalization**: Learning for each state is independent
- **Memory Requirements**: Must store all state-action pairs

### Further Reading

To overcome these limitations, explore:
- **Deep Q-Network (DQN)**: Use neural networks instead of Q-Table → See `2_Cart_Pole_DQN/`
- **Policy Gradient Methods**: Learn policy directly instead of value function
- **Actor-Critic Methods**: Combine value and policy → See `3_Pendulum/`

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 6: Temporal-Difference Learning)
- Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards" (Ph.D. thesis)
