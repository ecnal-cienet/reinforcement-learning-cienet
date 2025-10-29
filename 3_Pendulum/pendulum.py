#!/usr/bin/env python

# pendulum.py
# 定義 Pendulum 環境的 Actor-Critic 網路結構

# pendulum詳細說明：
# 在這個任務中，目標是讓一個擺錘（pendulum）保持直立位置。擺錘可以通過施加一個力矩來控制。狀態空間包括擺錘的角度和角速度，而動作空間則是施加的力矩大小。獎勵函數旨在鼓勵擺錘保持直立並最小化施加的力矩。


import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import gymnasium as gym

# 為了讓 Actor 輸出 `sigma`，我們需要 TanhAF (雙曲正切)
# 確保標準差 (sigma) 永遠是正數。
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# --- 1. 定義「評論家 (Critic)」網路 ---
#   它的工作：學習 V(s)
class Critic(nnx.Module):
    def __init__(
            self,
            in_features: int,
            *,
            rngs: nnx.Rngs
        ) -> None:
        """
        建立評論家網路 (Critic Network)
        Args:
            in_features (int): 狀態空間的維度 (state dimension)
            rngs (nnx.Rngs): 用於初始化網路權重的隨機數生成器
        """
        self.fc1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 64, rngs=rngs)
        self.fc_out = nnx.Linear(64, 1, rngs=rngs) # 輸出 1 個數字 (V 值)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        前向傳播 (Forward Pass)
        Args:
            x (jax.Array): 狀態輸入 (state input)
        Returns:
            jax.Array: 狀態價值 V(s) 的估計值
        """
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        # 直接輸出 V(s) 的估計值
        return self.fc_out(x)

# --- 2. 定義「演員 (Actor)」網路 ---
#   它的工作：學習 π(a|s)
class Actor(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        """
        一個 MLP, 但有兩個「頭」(_mu 和 _sigma):
        Input(3) -> 64 -> ReLU -> 64 -> ReLU -> Output_mu(1)
                                             -> Output_sigma(1)
        Args:
            in_features (int): 狀態空間的維度 (state dimension)
            out_features (int): 動作空間的維度 (action dimension)
            rngs (nnx.Rngs): 用於初始化網路權重的隨機數生成器
        """
        self.fc1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 64, rngs=rngs)
        # 「平均值」頭
        self.fc_mu = nnx.Linear(64, out_features, rngs=rngs)
        # 「標準差」頭
        self.fc_sigma = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> tfd.Normal:
        """
        前向傳播 (Forward Pass)
        Args:
            x (jax.Array): 狀態輸入 (state input)
        Returns:
            tfd.Normal: 動作的常態分佈 (Normal distribution of actions)
        """
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))

        # 1. 計算平均值 (mu)
        # Pendulum 動作範圍是 [-2, 2]，所以我們用 tanh(雙曲正切) 把它縮放到 [-1, 1]
        # 再乘以 2 (環境的動作上限)
        mu = jnp.tanh(self.fc_mu(x)) * 2.0

        # 2. 計算標準差 (sigma)
        # sigma 必須是正數，所以我們用 softplus
        sigma = nnx.softplus(self.fc_sigma(x)) + 1e-5 # (加 1e-5 避免為 0)

        # 3. 回傳一個「機率分佈」
        # 我們不是回傳「動作」，而是回傳一個「常態分佈」物件
        return tfd.Normal(loc=mu, scale=sigma)

class PPOAgent:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            *,
            rng_key: jax.Array,
        ):
        """
        建立 PPO Agent 的 Actor-Critic 網路和優化器
        Args:
            state_dim (int): 狀態空間的維度 (state dimension)
            action_dim (int): 動作空間的維度 (action dimension)
            rng_key (jax.Array): 用於初始化網路權重的隨機數生成器
        """
        # 建立主 RNG 密鑰
        actor_key, critic_key = jax.random.split(rng_key)

        # --- 1. 建立 Actor (演員) 和它的優化器 ---
        self.actor = Actor(state_dim, action_dim, rngs=nnx.Rngs(actor_key))

        # PPO 通常使用較低的學習率
        self.actor_optimizer = nnx.Optimizer(
            self.actor,
            optax.adam(learning_rate=3e-4)
        )

        # --- 2. 建立 Critic (評論家) 和它的優化器 ---
        self.critic = Critic(state_dim, rngs=nnx.Rngs(critic_key))

        self.critic_optimizer = nnx.Optimizer(
            self.critic,
            optax.adam(learning_rate=1e-3)
        )

        # 我們還需要一個 RNG 密鑰流 (stream) 來處理採樣
        self.rng_stream = nnx.Rngs(jax.random.PRNGKey(42))

    def select_action(self, state: np.ndarray):
        """
        這是在「收集 (Rollout)」階段呼叫的函式。
        它會回傳「動作」、「V值」和「Log機率」。
        """
        # 將 numpy 狀態轉換為 jax array
        state_jnp = jnp.asarray(state[np.newaxis, :], dtype=jnp.float32)

        # 1. 呼叫 Actor 取得「機率分佈」
        action_dist = self.actor(state_jnp)

        # 2. 呼叫 Critic 取得「V 值 (基線)」
        #    我們用 jax.lax.stop_gradient 阻止梯度流向 Critic，
        #    因為我們在「採樣」時不想訓練 Critic
        value = self.critic(state_jnp)
        value = jax.lax.stop_gradient(value)

        # 3. 採樣一個動作
        rng_key = self.rng_stream.sampler() # 取得一個新密鑰
        action = action_dist.sample(seed=rng_key)

        # 4. 計算這個動作的「Log 機率」
        log_prob = action_dist.log_prob(action)

        # 5. 將 JAX arrays 轉換回 NumPy (以便存儲)
        action = np.asarray(action).flatten()
        log_prob = np.asarray(log_prob).flatten()
        value = np.asarray(value).flatten()

        # 我們回傳所有需要儲存的東西
        return action, value, log_prob

    def train_step(
            self,
            batch_states: jax.Array,
            batch_actions: jax.Array,
            batch_log_probs_old: jax.Array,
            batch_advantages: jax.Array,
            batch_returns: jax.Array,
            clip_epsilon: float, # 這就是超參數 EPSILON
        ):
        """
        這是在「學習 (Learn)」階段的核心。
        它會被反覆呼叫。
        """
        # --- 1. 訓練「評論家 (Critic)」 ---
        # Critic 的目標：讓 V(s) 盡可能接近「實際總分 (Returns)」
        def critic_loss_fn(critic_model: Critic):
            # (1) 取得「當前的 V 值預測」
            values_pred = critic_model(batch_states)
            # (2) 計算 V(s) 和「實際總分 (Returns)」之間的均方誤差 (MSE)
            loss = jnp.mean((batch_returns - values_pred.flatten())**2)
            return loss

        # (3) 計算梯度並更新 Critic
        _, critic_grads = nnx.value_and_grad(critic_loss_fn)(self.critic)
        self.critic_optimizer.update(critic_grads)

        # --- 2. 訓練「演員 (Actor)」 ---
        # Actor 的目標：最大化「Advantage」，但要被「Clipping」限制住
        def actor_loss_fn(actor_model: Actor):
            # (1) 取得「當前的機率分佈」和「新的 Log 機率」
            action_dist_new = actor_model(batch_states)
            log_probs_new = action_dist_new.log_prob(batch_actions)

            # (2) 計算「策略比例 (Policy Ratio)」
            #     Ratio = π_new / π_old
            #     在 log 空間中 = log(π_new) - log(π_old)
            #     exp(...) 把它轉回正常空間
            ratio = jnp.exp(log_probs_new - batch_log_probs_old)

            # (3) 計算「未裁剪的」Loss
            #     這就是 REINFORCE 的邏輯： Advantage * Ratio
            loss_unclipped = batch_advantages * ratio

            # (4) PPO 核心：計算「裁剪後的 (Clipped)」Loss
            #     jnp.clip 會把 Ratio 限制在 [1 - ε, 1 + ε] 之間
            ratio_clipped = jnp.clip(
                ratio,
                1.0 - clip_epsilon,
                1.0 + clip_epsilon
            )
            loss_clipped = batch_advantages * ratio_clipped

            # (5) 取「兩者中較小」的那個
            #     這就是 PPO 的「悲觀」原則：
            #     我只取「最保守」的那個更新訊號，防止「用力過猛」
            loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))

            # (我們加負號 -，因為優化器 (Adam) 只能「最小化」Loss，
            #  而我們想「最大化」Advantage)

            return loss

        # (6) 計算梯度並更新 Actor
        _, actor_grads = nnx.value_and_grad(actor_loss_fn)(self.actor)
        self.actor_optimizer.update(actor_grads)


# --- 3. 建立「Rollout 儲存區」 ---
#   它的工作：儲存 N 步的 (s, a, r, log_prob, v_val, done)
class RolloutBuffer:
    def __init__(self):
        """
        建立一個空的儲存區
        """
        # 我們用簡單的 Python 列表來儲存
        self.clear()

    def add(self, state, action, reward, log_prob, value, done):
        """
        將一個時間步的資料加入儲存區
        Args:
            state (np.ndarray): 狀態
            action (np.ndarray): 動作
            reward (float): 獎勵
            log_prob (float): 動作的 Log 機率
            value (float): 狀態價值 V(s)
            done (bool): 是否結束
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """清空儲存區"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        # 這兩個是我們「下一步」要計算的「學習目標」
        self.advantages = []
        self.returns = []

    def get_data_for_learning(self):
        """
        將儲存的資料轉換為 JAX 陣列，為「學習」做準備
        Returns:
            Tuple of jax.Array: 包含 states, actions, log_probs, advantages, returns
        """
        # (這裡的 np.asarray 確保了資料是一致的 NumPy 陣列)
        return (
            jnp.asarray(self.states, dtype=jnp.float32),
            jnp.asarray(self.actions, dtype=jnp.float32),
            jnp.asarray(self.log_probs, dtype=jnp.float32),
            jnp.asarray(self.advantages, dtype=jnp.float32),
            jnp.asarray(self.returns, dtype=jnp.float32)
        )

    def __len__(self):
        return len(self.states)

    def calculate_advantages_and_returns(
            self,
            last_value: np.ndarray,
            gamma: float,
            gae_lambda: float,
        ):
        """
        在 N 步收集完畢後，從「最後一步」反向計算 GAE 和 Returns。

        Args:
            last_value (np.ndarray):
                Critic 對「第 N+1 步」的 V 值預測 (因為第 N 步還沒結束)。
            gamma (float): 折扣因子 (e.g., 0.99)
            gae_lambda (float): GAE 的平滑參數 (e.g., 0.95)
        """
        # 我們需要一個 `next_value` 來啟動迴圈
        next_value = last_value
        # 我們將從後往前填充 advantages 和 returns
        self.advantages = [0] * len(self.rewards)
        self.returns = [0] * len(self.rewards)

        # 迴圈從「最後一步 (N-1)」跑到「第一步 (0)」
        for t in reversed(range(len(self.rewards))):
            # 獲取第 t 步的資料
            reward = self.rewards[t]
            value = self.values[t]
            done = self.dones[t]

            # --- 1. 計算 TD-Error (delta) ---
            # 這是 Critic 的「一步之差」的驚訝程度
            # delta = (當下獎勵 + γ * 下一步的V值) - (當下的V值)
            # 如果 (done=True)，下一步的V值為 0
            delta = reward + gamma * next_value * (1.0 - done) - value

            # --- 2. 計算 GAE (Advantage) ---
            # Advantage(t) = delta(t) + (γ * λ) * Advantage(t+1)
            # 這是一個遞迴：t 時刻的優勢 =
            #   (t 時刻的驚訝) + (折扣後的 *下一步的優勢*)
            gae = delta + gamma * gae_lambda * (1.0 - done) * (self.advantages[t+1] if t+1 < len(self.rewards) else 0.0)
            self.advantages[t] = gae

            # --- 3. 計算 Returns (Critic 的學習目標) ---
            # Return(t) = Advantage(t) + Value(t)
            # (因為 A(t) = R(t) - V(t)，所以 R(t) = A(t) + V(t))
            self.returns[t] = gae + value

            # 更新下一步的 V 值，用於計算 t-1 步的 delta
            next_value = value

        # [PPO 優化]：標準化 Advantage
        # 這是 PPO 穩定訓練的另一個關鍵技巧
        # 我們讓 Advantage 的平均值為 0，標準差為 1
        # 這可以防止「獎勵訊號」忽大忽小
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8 # (加 1e-8 避免除以 0)
        self.advantages = (self.advantages - adv_mean) / adv_std

# --- 4. 主訓練迴圈 ---
def main():
    """
    PPO 主訓練函式
    """
    print("開始 PPO 訓練...")

    # --- A. 初始化 ---
    env = gym.make("Pendulum-v1")

    # 取得狀態和動作的維度
    STATE_DIM = env.observation_space.shape[0] # 3
    ACTION_DIM = env.action_space.shape[0]     # 1

    # 初始化 Agent
    main_rng = jax.random.PRNGKey(42)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, rng_key=main_rng)

    # 初始化 Rollout 儲存區
    buffer = RolloutBuffer()

    # --- B. 定義超參數 ---
    NUM_TOTAL_TIMESTEPS = 100_000 # 總共要跑的步數
    ROLLOUT_STEPS = 2048          # 每次「收集」的步數 (N)
    TRAIN_EPOCHS = 10             # 每次「學習」要反覆訓練幾次
    BATCH_SIZE = 64               # 每次訓練抓的小批次大小

    GAMMA = 0.99                  # 折扣因子
    GAE_LAMBDA = 0.95             # GAE 的平滑參數 (λ)
    CLIP_EPSILON = 0.2            # PPO 的裁剪參數 (ε)

    # --- C. PPO 的「收集-學習」大迴圈 ---
    # 初始化環境
    state, _ = env.reset(seed=42)
    current_total_steps = 0

    while current_total_steps < NUM_TOTAL_TIMESTEPS:
        print(f"\n--- 正在收集 {ROLLOUT_STEPS} 步的資料 ---")

        # --- 階段 1: 收集 (Rollout) ---
        for _ in range(ROLLOUT_STEPS):
            # 1. 呼叫 Agent 的 select_action
            #    取得「動作」、「V值基線」和「舊 Log 機率」
            action, value, log_prob = agent.select_action(state)

            # 2. 與環境互動
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 將這一步的經驗存入儲存區
            buffer.add(state, action, reward, log_prob, value, done)

            # 4. 更新狀態
            state = next_state
            current_total_steps += 1

            # 如果遊戲結束 (done=True)，重置環境
            if done:
                state, _ = env.reset()

        # --- 階段 2: 計算學習目標 (Advantage & Returns) ---

        # 我們需要「最後一步 (N+1)」的 V 值來啟動 GAE 計算
        # 我們用 Critic 預測「最後」的 next_state
        last_value_jnp = agent.critic(jnp.asarray(state[np.newaxis, :]))
        last_value = np.asarray(last_value_jnp).flatten()

        # 計算 GAE (Advantages) 和 Returns
        print("...正在計算 GAE (Advantages) 和 Returns...")
        buffer.calculate_advantages_and_returns(last_value, GAMMA, GAE_LAMBDA)

        # --- 階段 3: 學習 (Learn) ---
        print(f"...開始 {TRAIN_EPOCHS} 個 Epochs 的學習...")

        # 取得所有「準備好」的學習資料
        (
            all_states,
            all_actions,
            all_log_probs_old,
            all_advantages,
            all_returns
        ) = buffer.get_data_for_learning()

        # 我們要拿著這 2048 筆資料，反覆訓練 TRAIN_EPOCHS 次
        for _ in range(TRAIN_EPOCHS):
            # 為了穩定訓練，我們把資料打亂 (Shuffle)
            indices = jax.random.permutation(agent.rng_stream.sampler(), ROLLOUT_STEPS)

            # 把 2048 筆資料切分成多個 BATCH_SIZE (64) 的小批次
            for start in range(0, ROLLOUT_STEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                # 取得這一小批次的資料
                batch_states = all_states[batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_log_probs_old = all_log_probs_old[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_returns = all_returns[batch_indices]

                # ** 呼叫 PPO 的核心引擎 **
                agent.train_step(
                    batch_states,
                    batch_actions,
                    batch_log_probs_old,
                    batch_advantages,
                    batch_returns,
                    CLIP_EPSILON
                )

        # --- 階段 4: 丟棄 (Discard) ---
        # 學習完畢！丟棄所有「舊」資料
        buffer.clear()

        # 顯示進度
        print(f"目前總步數: {current_total_steps}/{NUM_TOTAL_TIMESTEPS}")

    env.close()
    print("--- 訓練完成！ ---")

# 執行主函式
if __name__ == "__main__":
    main()
