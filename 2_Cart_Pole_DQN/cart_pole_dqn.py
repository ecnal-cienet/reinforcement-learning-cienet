#!/usr/bin/env python
"""
DQN Agent for CartPole-v1 using Flax NNX and Optax.
"""

import random
from collections import deque

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# --- 1. 超參數 (Hyperparameters) ---
STATE_DIM = 4
ACTION_DIM = 2
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
NUM_EPISODES = 500
TARGET_UPDATE_FREQ = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# --- 2. 使用 Flax NNX 定義 Q-Network ---
class QNetwork(nnx.Module):
    """
    DQN 的神經網路模型 (函數近似器).
    """
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        """
        定義網路結構: Input(4) -> 64 -> 64 -> Output(2)
        """
        self.fc1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 64, rngs=rngs)
        self.fc3 = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        定義前向傳播.
        參數:
            x: 輸入狀態 (state)
        回傳:
            輸出各動作的 Q 值 (logits).
        範例：
            logits = model(jnp.array([[0.0, 0.0, 0.0, 0.0]]))
            print(logits)
            # jax.Array([[0.1, -0.2]])
        """
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)  # 輸出原始 Q 值 (Logits)

# --- 3. 經驗回放 (Replay Buffer) ---
class ReplayBuffer:
    """
    用於儲存和抽樣 (s, a, r, s', done) 經驗的記憶體.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done): # pylint: disable=too-many-arguments
        """
        將一筆經驗存入記憶體.
        """
        self.buffer.append((
            np.expand_dims(state, 0),
            action,
            reward,
            np.expand_dims(next_state, 0),
            done
        ))

    def sample(self, batch_size: int):
        """
        從記憶體中隨機抽樣一個 batch 的經驗.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.concatenate(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.concatenate(next_states),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)

# --- 4. DQN Agent (最核心的類別) ---
# pylint: disable=too-many-instance-attributes
class DQNAgent:
    """
    DQN Agent, 封裝了神經網路、優化器、經驗回放和訓練邏輯.
    """
    def __init__(self, state_dim, action_dim, *, rng_key):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.total_steps = 0

        # Create NNX RNG streams from the key
        online_key, target_key = jax.random.split(rng_key)

        # 1. 建立「線上網路 (Online Network)」
        self.online_network = QNetwork(state_dim, action_dim, rngs=nnx.Rngs(online_key))

        # 2. 建立「目標網路 (Target Network)」
        self.target_network = QNetwork(state_dim, action_dim, rngs=nnx.Rngs(target_key))

        # Use NNX Optimizer wrapper for optax optimizer
        self.optimizer = nnx.Optimizer(self.online_network, optax.adam(LEARNING_RATE))

    def select_action(self, state, rng_key: jax.Array) -> jax.Array:
        """
        使用 Epsilon-Greedy (E&E) 策略選擇動作.
        """
        if np.random.rand() <= self.epsilon:
            # 探索：隨機選擇
            return jax.random.randint(rng_key, (), 0, self.action_dim)

        # 利用：
        state_jnp = jnp.asarray(state[np.newaxis, :], dtype=jnp.float32)
        q_values = self.online_network(state_jnp)
        return jnp.argmax(q_values)

    def update_epsilon(self):
        """
        更新 Epsilon (Epsilon 衰退).
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        """
        將 Online Network 的權重「複製」到 Target Network.
        """
        print("...同步 Target Network 權重...")
        # Extract state from online network and update target network
        online_state = nnx.state(self.online_network)
        nnx.update(self.target_network, online_state)

    def train_step(self):
        """
        DQN 最核心的訓練步驟.
        """
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states = jnp.asarray(states)
        actions = jnp.asarray(actions, dtype=jnp.int32)
        rewards = jnp.asarray(rewards)
        next_states = jnp.asarray(next_states)
        dones = jnp.asarray(dones)

        # 2. 計算「TD 目標」(使用 Target Network 來固定靶心)
        q_next_target = self.target_network(next_states)
        q_next_max = jnp.max(q_next_target, axis=1)
        td_target = rewards + GAMMA * q_next_max * (1.0 - dones)

        # (3) 定義 Loss 函數 (使用 NNX 的方式)
        def loss_fn(model: QNetwork):
            """
            計算單一批次的 Loss.
            """
            q_current = model(states)
            q_actions = actions[:, None]
            q_current_action_batch = jnp.take_along_axis(q_current, q_actions, axis=1)
            q_current_action = jnp.squeeze(q_current_action_batch)
            loss = jnp.mean((q_current_action - td_target) ** 2)
            return loss

        # (4) 計算梯度並更新
        _, grads = nnx.value_and_grad(loss_fn)(self.online_network)
        self.optimizer.update(grads)

# --- 5. 訓練迴圈 ---
# pylint: disable=too-many-locals
def main():
    """
    主訓練函式.
    """
    print("開始訓練 DQN Agent...")
    env = gym.make("CartPole-v1")

    main_rng = jax.random.PRNGKey(42)
    agent_rng, training_rng = jax.random.split(main_rng)

    # 建立 DQN Agent
    agent = DQNAgent(STATE_DIM, ACTION_DIM, rng_key=agent_rng)

    total_rewards = []

    # 開始訓練迴圈
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # 1. 決定動作 (E&E)
            training_rng, action_rng = jax.random.split(training_rng)
            action = agent.select_action(state, action_rng)

            # 2. 與環境互動
            action_int = int(action)
            next_state, reward, terminated, truncated, _ = env.step(action_int)
            done = terminated or truncated

            # 3. 儲存經驗到 Replay Buffer
            agent.buffer.add(state, action, reward, next_state, done)

            # 4. 訓練 (更新) Online Network
            agent.train_step()

            state = next_state
            episode_reward += reward
            agent.total_steps += 1

            # 5. 定期更新 Target Network (固定靶心)
            if agent.total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

        # 6. Epsilon 衰退
        agent.update_epsilon()

        total_rewards.append(episode_reward)
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            print(f"Episode {episode + 1}, Epsilon: {agent.epsilon:.3f}, "
                  f"Avg Reward (last 50): {avg_reward:.2f}")

    env.close()
    print("訓練完成！")

if __name__ == "__main__":
    main()
