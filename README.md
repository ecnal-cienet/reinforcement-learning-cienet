# Reinforcement Learning (RL) - Complete Learning Guide

This comprehensive guide covers everything from core RL concepts to Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO), with complete theory and hands-on implementations.

> **中文版本**: [README_ch.md](README_ch.md)

## Table of Contents

0.  [Installation Guide](#installation-guide)
1.  [Module 1: Foundations of RL](#module-1-the-foundations-of-rl)
    * 1.1 Core RL Loop (Agent, Environment, S, A, R)
    * 1.2 Markov Decision Processes (MDPs) & Markov Property
    * 1.3 Policy & Value Function
    * 1.4 Exploration vs. Exploitation
2.  [Module 2: Tabular Methods (Q-Learning)](#module-2-tabular-methods-q-learning)
    * 2.1 Q-Function
    * 2.2 Temporal-Difference (TD) Learning
    * 2.3 Q-Learning Algorithm (Q-Table Updates)
    * 📁 [Implementation: `1_Q_Learning/Q_Learning.py`](1_Q_Learning/Q_Learning.py)
3.  [Module 3: Deep Q-Networks (DQN)](#module-3-deep-q-networks-dqn)
    * 3.1 Function Approximation
    * 3.2 DQN Key Techniques (Experience Replay & Target Network)
    * 3.3 DQN Implementation (Flax NNX)
    * 📁 [Implementation: `2_Cart_Pole_DQN/cart_pole_dqn.py`](2_Cart_Pole_DQN/cart_pole_dqn.py)
4.  [Module 4: Policy Gradients](#module-4-policy-gradients-the-why)
    * 4.1 Limitations of DQN
    * 4.2 Policy Gradient (REINFORCE) Algorithm
    * 4.3 REINFORCE Problems (High Variance & Credit Assignment)
5.  [Module 5: Actor-Critic & PPO](#module-5-actor-critic--ppo-the-how)
    * 5.1 Baseline & Advantage
    * 5.2 Actor-Critic Architecture
    * 5.3 PPO (Proximal Policy Optimization) Core Theory
    * 5.4 PPO Implementation (Flax NNX)
    * 📁 [Implementation: `3_Pendulum/pendulum.py`](3_Pendulum/pendulum.py)
6.  [Module 6: Distributed Training with JAX/Maxtext](#module-6-distributed-training-with-jaxmaxtext)
    * 6.1 The Memory Bottleneck
    * 6.2 Key Concept 1: `Mesh` (Hardware Mapping)
    * 6.3 Key Concept 2: Two Parallelism Strategies
    * 6.4 Key Concept 3: `logical_axis_rules` (Sharding Rules)
7.  [Module 7: GRPO (Group Relative Policy Optimization)](#module-7-grpo-group-relative-policy-optimization)
    * 7.1 GRPO Core Trade-offs
    * 7.2 Project Implementation: GRPO on MNIST
    * 📁 [Implementation: `4_GRPO_MNIST/grpo_mnist.py`](4_GRPO_MNIST/grpo_mnist.py)

---

## <a name="installation-guide"></a>Installation Guide

This project uses Python 3.10.0. We recommend using a virtual environment to manage dependencies.

### Step 1: Create Virtual Environment (if not already created)

```bash
python3 -m venv .venv
```

### Step 2: Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

All required Python packages are listed in `requirements.txt`. Install them with a single command:

```bash
pip install -r requirements.txt
```

**Main dependencies include:**
- **numpy**: Array operations
- **jax + jaxlib**: High-performance numerical computing
- **flax**: JAX neural network framework (NNX API)
- **optax**: Optimization algorithms
- **gymnasium**: RL environments (CartPole-v1, Pendulum-v1)
- **tensorflow-probability[jax]**: Probability distributions (for continuous action spaces)

### Step 4: Verify Installation

After installation, verify your setup by running:

```bash
# Run Q-Learning implementation
python 1_Q_Learning/Q_Learning.py

# Run DQN implementation
python 2_Cart_Pole_DQN/cart_pole_dqn.py

# Run PPO implementation
python 3_Pendulum/pendulum.py

# Run GRPO implementation
python 4_GRPO_MNIST/grpo_mnist.py
```

---

## <a name="module-1-the-foundations-of-rl"></a>Module 1: Foundations of RL

This module introduces the basic "worldview" and common vocabulary of RL.

### 1.1 The Core RL Loop

Everything in RL is based on the interaction loop between an **Agent** and an **Environment**.

1.  **Agent**: The learner or decision-maker (e.g., game character).
2.  **Environment**: The external world the agent interacts with (e.g., game level).
3.  **State (S)**: A snapshot of the environment at a given moment (e.g., character's (x, y) coordinates).
4.  **Action (A)**: The choices the agent can make given state S (e.g., "move left").
5.  **Reward (R)**: Feedback signal from the environment after executing action A (e.g., collect coin `+10`, hit enemy `-50`).

**The agent's sole objective**: Maximize **cumulative future reward**.

### 1.2 Markov Decision Processes (MDPs)

MDPs are the mathematical framework for describing the RL loop. The core assumption is the **Markov Property**.

> **Markov Property (Memoryless Property)**:
> The future state depends **only** on the "current state" and "current action," not on "how we arrived at the current state."

In other words, **"the current state S" already contains all the historical information needed for decision-making**.
* **Satisfies:** Go (the current board position is everything).
* **Doesn't satisfy:** Poker (you need to remember opponents' past betting behavior).

### 1.3 Policy & Value Function

These two concepts describe the agent's "brain" and "goal."

* **Policy (π)**:
    * The agent's "behavioral rules" or "decision brain."
    * It's a function that determines which action A to take given state S.
    * π(A | S) = probability of executing action A in state S.
    * **Our goal**: Find the "**optimal policy (π*)**" to achieve maximum total reward.

* **Value Function (V(s))**:
    * An "evaluation function" to measure "**how good a state S is**."
    * V(s) = "Starting from state S and following policy π to the end, the **expected** total future reward."
    * The agent uses V(s) to improve π (e.g., choose the action A that leads to a "higher value V(s')" state).

### 1.4 Exploration vs. Exploitation

This is the core dilemma agents face during learning.

* **Exploitation**:
    * **Definition**: Based on **current knowledge**, make the "best" choice.
    * **Example**: Go to your favorite restaurant (you know it's 90 points).
* **Exploration**:
    * **Definition**: **Intentionally** try some "unknown" choices to "collect new information."
    * **Example**: Try a new restaurant (might be 10 points, might be 100 points).

A good agent must balance these two. **Epsilon-Greedy (ε-Greedy)** is the most common strategy:
* With probability (1-ε) (e.g., 90%), "exploit."
* With probability ε (e.g., 10%), "explore."

---

## <a name="module-2-tabular-methods-q-learning"></a>Module 2: Tabular Methods (Q-Learning)

This module introduces the first concrete RL algorithm, suitable for problems with very small state spaces (e.g., 4x4 grid).

### 2.1 Q-Function

Q-Learning introduces a **Q-function** that's more powerful than V(s), also called **Q-value**.

* V(s): How good is "being in state s"?
* Q(s, a): How good is "being in state s **and** executing action a"?

The Q-function is more direct. When the agent is in state s, it doesn't need to think about V(s'); it just compares:
* Q(s, left) = 10
* Q(s, right) = 50
* ...then choose the action with the highest Q-value (right).

In tabular methods, we use a **Q-Table** (e.g., NumPy array) to store the Q-value for **every** (s, a) pair.

### 2.2 Temporal-Difference (TD) Learning

Q-Learning is a **Temporal-Difference (TD) Learning** method.

* **Core idea**: We **don't need** to wait until the game ends to update values. We "**use future reality to correct past estimates**" at **every step**.
* **Example**: You estimate the commute takes 30 minutes. After 5 minutes, you see the highway is jammed. You **immediately** (TD Learning) update your estimate (to 2 hours), rather than "learning" this 2 hours later.

### 2.3 Q-Learning Algorithm (Q-Table Updates)

Q-Learning's goal is to learn this Q-Table. After the agent executes a `(S, A, R, S')` experience, it uses "TD Learning" to update the Q-Table:

**1. Calculate "TD Target" (i.e., "more accurate reality")**
$$
\text{TD Target} = R + \gamma \cdot \max_{a'} Q(S', a')
$$
* **R**: Reward you **immediately** received.
* **γ (gamma)**: Discount factor (e.g., 0.99), representing how much you value future rewards.
* **max<sub>a'</sub> Q(S', a')**: The agent checks the Q-Table and finds the Q-value of the "**best next action**" in "**new state S'**".

**2. Calculate "TD Error"**
$$
\text{TD Error} = \text{TD Target} - Q(S, A)
$$
* TD Error is the difference between "new reality" and "old estimate."

**3. Update Q-Table**
$$
Q(S, A) \leftarrow Q(S, A) + \alpha \cdot (\text{TD Error})
$$
* **α (alpha)**: Learning rate (e.g., 0.1), representing how much to "trust" this error.
* **Logic**: Move the "old Q-value" **a little bit** towards the "more accurate TD Target."

**Implementation (NumPy)**:
We implement this algorithm in a 4x4 grid world using NumPy array `q_table = np.zeros((16, 4))`.

📁 **Complete implementation**: [`1_Q_Learning/Q_Learning.py`](1_Q_Learning/Q_Learning.py)
📖 **Detailed documentation**: [`1_Q_Learning/README.md`](1_Q_Learning/README.md)

---

## <a name="module-3-deep-q-networks-dqn"></a>Module 3: Deep Q-Networks (DQN)

This module marks the crucial leap from "tabular" methods to "deep learning" methods.

### 3.1 Function Approximation

**1. The "Table" Curse (Curse of Dimensionality)**
* **Problem**: Q-Tables (Module 2) only work for problems with **extremely small** state spaces (e.g., 16 states).
* **Example**: Playing Atari games, the state is an 84×84 pixel image. Total possible states (e.g., 4<sup>84×84</sup>) far exceed the number of atoms in the universe.
* **Conclusion**: We **cannot** build a Q-Table to "store" all Q-values.

**2. Solution: Function Approximation**
* **Core idea**: Instead of **storing** all Q-values, train an "**estimator**" (a function) to **estimate** Q-values.
* **Our estimator**: **Deep Neural Network**.

**3. Deep Q-Network (DQN)**
* DQN is a neural network trained to **play the role of a "Q-Table"**.
* **Input**: State S (e.g., game screen or CartPole's 4 numbers).
* **Output**: A vector representing Q-values for **all possible actions**.
    * `Q_Network(S)` → `[Q(S, a_1), Q(S, a_2), ...]`

**4. Benefits of Function Approximation**
1.  **Memory efficiency**: A network with a few million parameters can estimate Q-values for an "infinite" state space.
2.  **Generalization**:
    * In Q-Tables, states `(1,1)` and `(1,2)` are two **independent** entries.
    * In neural networks, `state (1,1)` and `state (1,2)` are **very similar** inputs.
    * Experience learned at `(1,1)` (e.g., "going right is good") automatically **generalizes** to `(1,2)`, helping it guess "going right at (1,2) might also be good."

### 3.2 DQN Key Techniques (Experience Replay & Target Network)

Directly training a neural network with Q-Learning's TD update formula is **extremely unstable**. DQN introduces two key techniques (stabilizers) to solve this problem.

**1. Stabilizer #1: Experience Replay**
* **Problem**: Neural network training hates "**highly correlated**" data. Using consecutive game experiences `(s_t, s_{t+1}, s_{t+2}, ...)` for training causes the network to "overfit" to the current game region and "**forget**" what it learned before.
* **Solution**: Build a "**Replay Buffer**" (a `deque`).
    1.  **Collect**: The agent plays the game normally, storing **every** step's experience `(S, A, R, S')` in the Replay Buffer (e.g., storing the last 100k steps).
    2.  **Train**: When training the network, we **don't use** "recent experiences," but **randomly sample** a mini-batch (e.g., 64) of **uncorrelated** old experiences from the Replay Buffer.
* **Benefits**: Breaks data correlation, making training more stable.

**2. Stabilizer #2: Target Network**
* **Problem**: The "**Moving Target**" problem.
* **Theory**: In Q-Learning updates, we use "one network" to compute both **"predicted value"** and **"target value"**.
    * TD Target = R + γ · max Q<sub>new</sub>(S', a')
    * Loss = (TD Target - Q<sub>new</sub>(S, A))<sup>2</sup>
* **Problem**: It's like you (the network) are aiming at a target, but the target is also determined by you (the network). Every time you adjust your stance (update weights), the target also moves randomly, and you can never aim accurately.
* **Solution**: Use **two** neural networks.
    1.  **Online Network (Q<sub>online</sub>)**:
        * This is the network we're **mainly** training.
        * It computes the "**predicted value**" Q<sub>online</sub>(S, A).
    2.  **Target Network (Q<sub>target</sub>)**:
        * This is a "**copy**" of the Online Network with **frozen** weights.
        * It **only** computes the "**target value**" TD Target = R + γ · max Q<sub>target</sub>(S', a').
* **Operation**:
    1.  `Online Network` (shooter) chases `Target Network` (**fixed target**), making training stable.
    2.  Every `N` steps (e.g., 1000 steps), we "sync": copy the **new weights** from `Online Network` to `Target Network` (move the target).

### 3.3 DQN Implementation (Flax NNX)

We implemented a DQN Agent to solve the "CartPole-v1" problem using Flax NNX.

* **`QNetwork(nnx.Module)`**: We built a 3-layer MLP with `nnx.Linear` as the function approximator.
* **`ReplayBuffer(deque)`**: We implemented `add()` and `sample()` methods.
* **`DQNAgent`**:
    * `__init__`: Initialized `online_network` and `target_network`.
    * **Key API (Flax NNX)**:
        * Use `nnx.Optimizer(model, optax.adam(...))` to **bind** the `optax` optimizer to the model.
        * Use `nnx.state(online_model)` to **extract** weights.
        * Use `nnx.update(target_model, online_state)` to **copy** weights (implement `Target Network` synchronization).
* **`train_step` (Training Step)**:
    1.  `sample()` a `batch` from `ReplayBuffer`.
    2.  **Compute Target**: `td_target = batch_rewards + GAMMA * jnp.max(self.target_network(batch_next_states), axis=1)`
    3.  **Compute Loss**: Define `loss_fn`, compute **Mean Squared Error (MSE)** between "predicted value" (from `self.online_network`) and `td_target`.
    4.  **Update**: Use `nnx.value_and_grad` and `self.optimizer.update(grads)` to update `online_network`.

📁 **Complete implementation**: [`2_Cart_Pole_DQN/cart_pole_dqn.py`](2_Cart_Pole_DQN/cart_pole_dqn.py)
📖 **Detailed documentation**: [`2_Cart_Pole_DQN/README.md`](2_Cart_Pole_DQN/README.md)

---

## <a name="module-4-policy-gradients-the-why"></a>Module 4: Policy Gradients

This module introduces RL's **second major family**: "Policy-Based" methods. This is the **direct ancestor** of `PPO/GRPO` algorithms used in your work.

### 4.1 Limitations of DQN (Value-Based)

DQN (Module 3) is very powerful but has fatal flaws:
1.  **Cannot handle continuous actions**: DQN relies on the max<sub>a'</sub> operation to select actions. If the action is "steering wheel angle" (a continuous number from -180 to +180), you can't take max over "infinitely many" actions.
2.  **Not direct enough**: What we really want is the "policy" itself. DQN can only "indirectly" derive the policy from Q-values.

### 4.2 Policy Gradient Algorithm

**Core idea**: We **directly** learn the "Policy" itself.

We build a "**Policy Network**", denoted π<sub>θ</sub> (where θ represents neural network weights).
* **Input**: State S.
* **Output**: A **probability distribution** representing probabilities for all actions.
    * **Discrete (CartPole)**: `[P(left), P(right)]` (e.g., `[0.7, 0.3]`)
    * **Continuous (Llama LLM)**: `[P(word A), P(word B), ...]` (probabilities for all vocabulary)
    * **Continuous (Pendulum)**: **Parameters** of a probability distribution (e.g., mean μ=0.5, standard deviation σ=0.1).

**Learning Logic: REINFORCE (Monte Carlo Method)**
REINFORCE is the most basic policy gradient algorithm.
1.  **Play a Full Episode**: Let the **current** policy network π<sub>θ</sub> play from start to finish.
2.  **Calculate Total Score**: Compute the "**discounted total reward (R<sub>total</sub>)**" for this entire episode.
3.  **Review (Credit Assignment)**:
    * Review **every step** `(S, A)` in this episode.
    * **If R<sub>total</sub> is "good" (e.g., +200)**: We "**reward**" all actions taken in this episode, **increasing** the probability π<sub>θ</sub>(A|S).
    * **If R<sub>total</sub> is "bad" (e.g., +10)**: We "**penalize**" all actions taken in this episode, **decreasing** the probability π<sub>θ</sub>(A|S).

### 4.3 Fatal Flaws of REINFORCE

This simple algorithm has two fatal flaws that you discovered yourself:

1.  **High Variance / Credit Assignment Problem**:
    * REINFORCE is like a professor who "**only looks at the team's total score**."
    * An A+ team project (good game) will **also reward** "slacking" members (bad actions).
    * An F- team project (bad game) will **also punish** "hardworking" members (good actions).
    * This learning signal is **very noisy**.

2.  **No Baseline Problem**:
    * Your key question: "Why is `+200` points considered good?"
    * If the average score is `+50`, `+200` is an A+.
    * If the average score is `+400`, `+200` is an F-.
    * The "**absolute total score (R<sub>total</sub>)**" itself is a **meaningless** learning signal.

---

## <a name="module-5-actor-critic--ppo"></a>Module 5: Actor-Critic & PPO

This module is the core of modern RL. It **fixes** all the flaws of REINFORCE and leads directly to `PPO/GRPO` in your work.

### 5.1 Baseline & Advantage

To fix REINFORCE, we need a "**relative score**", not an "absolute score."

**1. Introduce "Baseline"**
* **Theory**: We need a "standard" to judge whether `+200` points is good or bad.
* **Best baseline**: The "**Value Function V(s)**" (from Module 1).
* V(s) tells us: "In state S, how much score do I **expect** to get?"

**2. Introduce "Advantage" (High-Quality Learning Signal)**
* We can now compute a high-quality "relative score" called "**Advantage (A<sub>t</sub>)**".
* Advantage = Actual score received - Critic's expected score
* A<sub>t</sub> = R<sub>t</sub> - V(s<sub>t</sub>)
* **Signal interpretation**:
    * **A<sub>t</sub> > 0 (e.g., `+20`)**: Great! Your performance was **20 points better than expected**. This is a **high-quality "reward" signal**.
    * **A<sub>t</sub> < 0 (e.g., `-10`)**: Oops! Your performance was **10 points worse than expected**. This is a **high-quality "penalty" signal**.

### 5.2 Actor-Critic Architecture

To get both "policy" and "baseline (V-value)" simultaneously, we need **two** neural networks:

1.  **The Actor (Policy Network π<sub>θ</sub>)**
    * **Job**: Make decisions (output action probabilities). (e.g., `policy_model`)
    * **Learning**: Uses "**Advantage (A<sub>t</sub>)**" signal to learn.

2.  **The Critic (Value Network V<sub>φ</sub>)**
    * **Job**: **Only** "scores," providing the baseline V(s<sub>t</sub>).
    * **Learning**: Uses "**TD Error**" to learn (make V(s<sub>t</sub>) as close to R<sub>t</sub> as possible).

**PPO/GRPO** are the latest members of the Actor-Critic family.

### 5.3 PPO (Proximal Policy Optimization) Core Theory

**New Problem with Actor-Critic**: Training instability. The Actor, based on Advantage from a "possibly still weak" Critic, updates "**too aggressively**" at once, causing the just-learned policy to **collapse**.

**PPO's Solution**: Add "**Safety Locks**" to limit the Actor's update steps.

**Safety Lock #1: PPO-Clip (Hyperparameter `EPSILON = 0.2`)**
* **Theory**: This is a "**hard limit**" clipping method.
* **Ratio**: Ratio = π<sub>new</sub>(a|s) / π<sub>old</sub>(a|s) (new policy probability / old policy probability)
* **Clip**: PPO **forces** `Ratio` to be limited within `[1 - EPSILON, 1 + EPSILON]` (e.g., `[0.8, 1.2]`).
* **Loss Function**: PPO takes the **minimum** between "unclipped reward" and "clipped reward."
    * `Loss = min(Adv * Ratio, Adv * clip(Ratio, 1-ε, 1+ε))`
* **Conclusion**: `EPSILON` ensures that even when the Actor sees a huge Advantage, it **cannot** update the policy by more than 20% at once.

**Safety Lock #2: KL Divergence Penalty (Hyperparameters `BETA` & `reference_model`)**
* **Theory**: This is a "**soft limit**" penalty method.
* **`reference_model`**: A snapshot of the "**old policy (π<sub>old</sub>)**".
* **`KL Divergence`**: KL(π<sub>new</sub> || π<sub>old</sub>), measuring "how much the new and old policies differ."
* **Loss Function**: Loss = (Advantage) - (BETA × KL Divergence)
* **Conclusion**: `BETA` is like a "**rubber band**". The Actor can update freely, but if it runs too far from `reference_model` (`KL` gets large), `BETA` applies a **penalty** to pull it back.

### 5.4 PPO Implementation (Flax NNX - Pendulum Project)

We finally implemented a complete PPO Agent.

**1. `Actor(nnx.Module)` (Actor)**
* To handle "continuous actions," the network outputs **two** heads: `mu` (mean) and `sigma` (standard deviation).
* The `__call__` function returns a `tfd.Normal` (Normal distribution) object.

**2. `Critic(nnx.Module)` (Critic)**
* A standard MLP, the `__call__` function returns **one** number (V-value V(s)).

**3. `PPOAgent(nnx.Module)`**
* **`__init__`**: Initializes **two** networks (`actor`, `critic`) and **two** `nnx.Optimizer`s.
* **`select_action` (JAX vs. NumPy Bridge)**:
    1.  `state` (numpy) → `state_jnp` (jax, add batch dimension).
    2.  Call both `actor(state_jnp)` and `critic(state_jnp)` simultaneously.
    3.  `dist = actor(...)` gets the probability distribution.
    4.  `action = dist.sample(...)` **samples** an action (exploration).
    5.  `log_prob = dist.log_prob(action)` gets the **"old Log probability (log π<sub>old</sub>)"** (PPO learning **must have**).
    6.  `value = critic(...)` gets the **"baseline (V(s))"** (GAE calculation **must have**).
    7.  Convert `action`, `log_prob`, `value` back to `numpy`.

**4. `RolloutBuffer` (On-Policy Storage)**
* **On-Policy**: PPO **cannot** use DQN's ReplayBuffer.
* PPO's data is "**perishable**", must be **completely discarded (`clear()`)** after learning once.
* **`add(...)`**: Stores N steps of `(s, a, r, done, log_prob, value)`.
* **`calculate_advantages_and_returns(...)` (GAE Calculation)**:
    * This is the most crucial math.
    * After collecting N steps, iterate **backwards (reversed)** recursively to compute:
    * **TD Error (Delta)**: δ<sub>t</sub> = r<sub>t</sub> + γV(s<sub>t+1</sub>) - V(s<sub>t</sub>)
    * **Advantage**: A<sub>t</sub> = δ<sub>t</sub> + γλA<sub>t+1</sub> (GAE formula)
    * **Returns**: R<sub>t</sub> = A<sub>t</sub> + V(s<sub>t</sub>) (Critic's "correct answer")
    * **Optimization**: Finally, **standardize** `advantages` `(adv - mean) / std` to make Actor learning more stable.

**5. `PPOAgent.train_step` (PPO Engine Room)**
* **`train_step`** is called **repeatedly** (K Epochs).
* **Train Critic**:
    * `values_pred = self.critic(batch_states)`
    * `loss = jnp.mean((values_pred - batch_returns) ** 2)` (Standard MSE Loss)
    * `grads` → `critic_optimizer.update(grads)`
* **Train Actor (PPO-Clip Loss)**:
    1.  **Get new probabilities**: `log_probs_new = self.actor(batch_states).log_prob(batch_actions)`
    2.  **Compute Ratio**: `ratio = jnp.exp(log_probs_new - batch_log_probs_old)`
    3.  **Compute Loss (Unclipped)**: `loss_unclipped = batch_advantages * ratio`
    4.  **Compute Loss (Clipped)**: `ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)`
    5.  **...** `loss_clipped = batch_advantages * ratio_clipped`
    6.  **Take minimum**: `loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))` (add `-` sign to "minimize")
    7.  **Update**: `grads` → `actor_optimizer.update(grads)`

**6. `main()` (PPO Main Loop)**
PPO's lifecycle is a "collect-learn-discard" loop:
```python
while True:
    # --- 1. Collect (Rollout) ---
    # Call agent.select_action() N times
    # Call buffer.add() N times

    # --- 2. Compute Targets (GAE) ---
    # Call buffer.calculate_advantages_and_returns()

    # --- 3. Learn ---
    # Call agent.train_step() K times (on mini-batches)

    # --- 4. Discard ---
    # Call buffer.clear()
```

📁 **Complete implementation**: [`3_Pendulum/pendulum.py`](3_Pendulum/pendulum.py)
📖 **Detailed documentation**: [`3_Pendulum/README.md`](3_Pendulum/README.md)

---

## <a name="module-6-distributed-training-with-jaxmaxtext"></a>Module 6: Distributed Training with JAX/Maxtext

This module answers a core question: "Why can't we train models like Llama 3.1 on a single computer? And how do JAX/Maxtext solve this problem?"

### 6.1 The Memory Bottleneck

**Question**: Why can't we train an 8 billion (8B) parameter model on a "single" GPU or TPU core?

**Answer**: Not enough **memory (HBM)**.

* **Model Weights**:
    * `8 billion` parameters × `2 bytes/parameter (bfloat16)` = **16 GB**
* **Gradients**:
    * Our "learning engine" (`nnx.value_and_grad`) needs to compute a gradient for **every** parameter.
    * 8 billion parameters = 8 billion gradients = **another 16 GB**
* **Optimizer State**:
    * The `optax.adam` we use is an "adaptive" optimizer.
    * It needs to store **two** additional "moment" values (m and v) for **every** parameter.
    * `8 billion` parameters × `2 moments` × `2 bytes/value` = **another 32 GB**

**Total (Minimum Requirement):**
$$
16 \text{ GB (model)} + 16 \text{ GB (gradients)} + 32 \text{ GB (Adam state)} = \text{at least 64 GB}
$$

**Conclusion**: This `64 GB` load **cannot** fit into a **single** chip with only `32 GB` or `80 GB` HBM (high-bandwidth memory) (because additional space is needed for computation).

**Solution**: We **must** "split" this huge load across **multiple** chips to execute.

---

### 6.2 Key Concept 1: `Mesh` (Hardware Mapping)

Before we "split" the model, we must first "describe" our hardware layout.

* **`Mesh` (Grid)**: This isn't a physical thing, but a "**logical map**" JAX uses to understand your hardware cluster.
* **Example**: You have `8` TPU cores.
    * **JAX default**: `[c1, c2, c3, c4, c5, c6, c7, c8]` (a 1D list)
    * **Your `Mesh`**: `[[c1, c2], [c3, c4], [c5, c6], [c7, c8]]` (a 4x2 2D grid)
* **`Mesh` Axes**:
    * `Maxtext` (e.g., `config_ref.mesh_axes`) names the "dimensions" of this 2D grid.
    * Axis 0 (length 4) → Named `'data'`
    * Axis 1 (length 2) → Named `'model'`
* **Conclusion**: `Mesh` creates a "hardware map." We now have a `(4, 2)` chip grid with two "**logical physical axes**": `data` axis and `model` axis.

---

### 6.3 Key Concept 2: Two Parallelism Strategies

With the "map (`Mesh`)", we have two main ways to distribute "work":

**1. Data Parallelism - ("Replicate" Strategy)**
* **Approach**: **Copy** the **complete 64GB** model to **every** chip.
* **Speedup**: Split "work (Batch)" into 8 parts, 8 chips **simultaneously** process 8 different data parts.
* **Pros**: Fast.
* **Cons**: **Completely doesn't solve the memory problem**.

**2. Model/Tensor Parallelism - ("Split" Strategy)**
* **Approach**: **Split** the **64GB** model into 8 pieces, each `8GB`. **Each** chip **only** stores its own `8GB` piece.
* **Speedup**: Chip 1 computes its `8GB`, "communicates" the result to Chip 2...
* **Pros**: **Perfectly solves the memory problem**.
* **Cons**: Extremely complex engineering, requires extensive inter-chip communication.

**JAX/Maxtext Strategy: Best of Both Worlds**
`Maxtext` uses `Mesh`'s **multi-dimensionality** (e.g., `('data', 'model')`) to **simultaneously** implement both types of parallelism:

1.  JAX sees the `'model'` axis (length 2), it performs "**Model Parallelism**":
    * "OK, I'll split the model weights **into 2 parts**, stored along the `'model'` axis."
2.  JAX sees the `'data'` axis (length 4), it performs "**Data Parallelism**":
    * "OK, I'll split the data batch **into 4 parts**, distributed along the `'data'` axis."

---

### 6.4 Key Concept 3: `logical_axis_rules` (Sharding Rules Manual)

**Question**: How does JAX know which layer in the model should use which splitting method?

**Answer**: `logical_axis_rules` (Sharding Rules Manual).

`logical_axis_rules` is a "**translation dictionary**" that translates **"model internal"** names into **"hardware (`Mesh`)"** names.

1.  **Model's Internal "Logical Axes"**:
    * `Maxtext` assigns "logical names" to **every** dimension of **every** tensor when building the model.
    * `(batch_size, sequence_length)` → `('batch', 'sequence')`
    * `(hidden_dim, mlp_dim)` → `('embed', 'mlp')`
    * `(vocab_size, hidden_dim)` → `('vocab', 'embed')`

2.  **Rules Manual (`config_policy.logical_axis_rules`)**:
    * This is a dictionary where you define "translation rules."
    * `{ 'Logical Axis Name': 'Physical Mesh Axis Name' }`

    ```python
    # Example Rules Manual (Rulebook)
    rules = {
        'batch': 'data',    # Tell JAX: split 'batch' dimension into 4 parts along 'data' axis
        'mlp':   'model',   # Tell JAX: split 'mlp' dimension into 2 parts along 'model' axis
        'embed': None,      # Tell JAX: 'embed' dimension (hidden_dim), don't split (replicate)
    }
    ```

3.  **`with nn_partitioning.axis_rules(...)` (Auto Builder)**:
    * This is the `with` block in your code.
    * **JAX automatically executes**:
        1.  Reads your "hardware map (`Mesh`)".
        2.  Reads your "rules manual (`logical_axis_rules`)".
        3.  When it creates `nnx.Linear` (logical axes `('embed', 'mlp')`), it looks up the manual:
            * `'embed'` → `None` (don't split)
            * `'mlp'` → `'model'` (split into 2 parts along `model` axis)
        4.  **JAX automatically** splits this weight **into 2 pieces**, implementing **"Model Parallelism"**.
        5.  When it sees "data (Batch)" (logical axes `('batch', 'sequence')`), it looks up the manual:
            * `'batch'` → `'data'` (split into 4 parts along `data` axis)
        6.  **JAX automatically** splits data **into 4 parts**, implementing **"Data Parallelism"**.

**Summary**: You (the developer) only need to define the "hardware map (`Mesh`)" and "sharding rules (`axis_rules`)", and `Maxtext` and `JAX` will automatically complete all the complex distributed training work for you.

---

## <a name="module-7-grpo-group-relative-policy-optimization"></a>Module 7: GRPO (Group Relative Policy Optimization)

This module is a **capstone project** that combines PPO's "safety lock" theory with a more efficient **Critic-less** baseline calculation method.

This **perfectly** corresponds to the core idea of modern LLM training (like `GRPO on Maxtext`): **Removing the `Critic` network to save memory**.

### 7.1 GRPO's Core Trade-off

**1. Standard PPO (Our Module 5 Implementation)**
* **Actor π<sub>θ</sub>**: A neural network.
* **Critic V<sub>φ</sub>**: **Another** neural network.
* **Advantage**: A<sub>t</sub> = (Actual total score R<sub>t</sub>) - (Critic's expected score V<sub>φ</sub>(s<sub>t</sub>))
* **Problem**: Training `Critic` (V<sub>φ</sub>) requires **additional massive memory** (model weights + gradients + Adam state), which is **unbearable** when training LLMs with 8 billion (8B) parameters.

**2. GRPO (Group Relative) "Revolution"**
* **GRPO's core idea**: We **completely remove the `Critic` network**.
* **How does GRPO compute "Baseline"?**
    * It doesn't **learn** the baseline, but **calculates** it dynamically.
    * It relies on a "**Group's**" **average performance**.
* **GRPO's "Relative Advantage"**:
    * Advantage = (Individual performance R<sub>i</sub>) - (Group average performance R̄)

**3. GRPO's "Cost Shift"**
* **Pros**:
    * **Memory savings**: Saves all the memory required for an 8B parameter `Critic` network.
* **Cons**:
    * **Higher collection cost**: To compute "group average," you must generate `G` times (`NUM_GENERATIONS`) responses for **the same** state S, making the "data collection (Rollout)" phase's computational cost G times higher.
* **Conclusion**: GRPO is an algorithm that **sacrifices "data collection" efficiency to gain "training" memory efficiency**.

---

### 7.2 Project Implementation: GRPO on MNIST

We applied this "LLM algorithm" to an "image classification" problem to verify its core logic.

**1. Framing "Classification" as RL**
* **State (s)**: A `(28, 28)` MNIST image (flattened to `(784,)`).
* **Action (a)**: The `Actor` network's **guess** for the digit (0-9).
* **This is a "One-Step" Episode**: The game ends after guessing.
* **Reward (r)**: Correct answer `+1.0`, wrong answer `0.0`.

**2. GRPO Core Implementation**
* **Actor Network**: `Input(784) → 128 → 128 → Output(10)`, outputs a `Categorical` distribution.
* **No Critic Network**: This is GRPO's key feature.
* **Group Baseline Calculation**:
    ```python
    # Generate predictions for a batch (G=1024) of images
    actions = actor.sample(batch_images)
    rewards = (actions == labels).astype(float)

    # Compute group average as baseline (replaces Critic)
    baseline = mean(rewards)

    # Compute relative advantage
    advantages = rewards - baseline
    ```
* **PPO-Clip Training**: Uses the same PPO-Clip Loss as Module 5 to train the Actor.

**3. Memory Savings**
* **PPO**: Needs Actor + Critic (two networks)
* **GRPO**: Only needs Actor (one network)
* **Savings**: About 50% of parameters (savings are even more significant in LLM scenarios)

**4. Expected Results**
* After 10 Epochs of training, accuracy should reach above 95%
* Proves that GRPO can be successfully applied to supervised learning problems

📁 **Complete implementation**: [`4_GRPO_MNIST/grpo_mnist.py`](4_GRPO_MNIST/grpo_mnist.py)
📖 **Detailed documentation**: [`4_GRPO_MNIST/README.md`](4_GRPO_MNIST/README.md)

---