# Proximal Policy Optimization (PPO) 實作 - Pendulum-v1

## 概述

這是一個使用 **Proximal Policy Optimization (PPO)** 演算法來解決 OpenAI Gymnasium 的 **Pendulum-v1** 環境的完整實作。PPO 是現代強化學習的核心演算法，被廣泛應用於機器人控制、遊戲 AI，以及**大型語言模型的對齊訓練 (RLHF)**。

**核心突破：** PPO 結合了 **Actor-Critic 架構**、**連續動作空間處理**、以及**穩定的策略更新機制**，是目前工業界最受歡迎的 RL 演算法之一。

**與前兩個專案的關係：**
- `1_Q_Learning/`: 學習 Q 值 (價值為基礎) - 表格型
- `2_Cart_Pole_DQN/`: 學習 Q 值 (價值為基礎) - 深度學習 + 離散動作
- `3_Pendulum/` **(本專案)**: 學習策略 (策略為基礎) - 深度學習 + **連續動作**

## 環境說明

### Pendulum-v1 (倒立擺)

Pendulum 是一個經典的連續控制問題：一個擺錘從隨機位置開始，目標是施加適當的力矩 (Torque) 來讓擺錘保持在**正上方**的直立位置。

```
    ↑ 目標位置
    |
    |
    O ← 樞紐 (Pivot)
   /
  /  ← 擺錘 (Pendulum)
 ●

目標：施加力矩讓擺錘旋轉到正上方並保持穩定
```

### 環境參數

- **狀態空間 (State Space)**：連續 3 維向量
  - `cos(θ)`: 擺錘角度的餘弦值 (範圍: -1 ~ 1)
  - `sin(θ)`: 擺錘角度的正弦值 (範圍: -1 ~ 1)
  - `θ̇`: 擺錘的角速度 (範圍: -8 ~ 8 rad/s)

  > **為什麼使用 cos/sin 而非角度？** 因為角度有週期性 (0° = 360°)，使用 cos/sin 可以讓狀態空間更平滑。

- **動作空間 (Action Space)**：**連續** 1 維向量
  - `torque`: 施加的力矩 (範圍: **-2 ~ 2**)
  - ⚠️ **關鍵差異**：這是**連續動作**，不像 CartPole 只有「左/右」兩個離散選擇

- **獎勵函數**：
  ```
  reward = -(θ² + 0.1 × θ̇² + 0.001 × torque²)
  ```
  - 懲罰擺錘偏離直立位置 (θ²)
  - 懲罰過大的角速度 (θ̇²)
  - 懲罰過大的力矩 (torque²，鼓勵節能)
  - **範圍**：約 -16.3 (最差) ~ 0 (完美)

- **終止條件**：
  - 沒有提前終止條件
  - 每個回合固定 200 步

- **成功標準**：
  - 平均獎勵 > -200 表示基本成功
  - 平均獎勵 > -150 表示良好控制

## 執行方式

### 前置條件

確保已啟動虛擬環境並安裝相依套件：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**特別注意：** 本專案需要 `tensorflow-probability` 來處理連續動作的機率分佈。

### 執行程式

```bash
python 3_Pendulum/pendulum.py
```

或者在 `3_Pendulum` 目錄內執行：

```bash
cd 3_Pendulum
python pendulum.py
```

## 演算法核心

### 從 DQN 到 PPO 的演進

#### DQN 的兩大局限

1. **無法處理連續動作空間**
   - DQN 依賴 `argmax` 操作：`action = argmax Q(s, a)`
   - 在 Pendulum 中，動作是 **-2.0 到 2.0 之間的任意實數**
   - 你無法對「無限多個」動作取 `max`

2. **只能間接學習策略**
   - DQN 學習的是「Q 值」，策略 π 是從 Q 值「推導」出來的
   - 我們真正想要的是「**策略 (Policy) 本身**」

#### PPO 的解決方案

**核心思想：** 直接學習一個「**策略網路 (Policy Network)**」，用 π_θ(a|s) 表示。

- **輸入**：狀態 s
- **輸出**：動作的**機率分佈** (而非單一動作)
  - **離散動作 (CartPole)**：`[P(左), P(右)]` = `[0.3, 0.7]`
  - **連續動作 (Pendulum)**：一個**常態分佈** `N(μ, σ²)`
    - `μ` (平均值)：最可能的動作
    - `σ` (標準差)：探索的程度

**範例：**
```python
state = [cos(θ), sin(θ), θ̇] = [0.8, 0.6, 1.2]
distribution = actor(state)  # → N(μ=1.5, σ=0.3)
action = distribution.sample()  # 從分佈中採樣 → 可能得到 1.7
```

### Actor-Critic 架構

PPO 使用**兩個**神經網路來協同工作：

#### 1. 演員 (Actor) - 策略網路 π_θ

**工作：** 決策者 (做動作)

**網路結構：** 雙頭 MLP
```
Input (3)  →  FC(64) → ReLU → FC(64) → ReLU → ┬→ FC_mu(1)    → tanh × 2 → μ
                                                 └→ FC_sigma(1) → softplus → σ
```

**關鍵設計：**
1. **μ 頭 (平均值)**：
   - 使用 `tanh` 將輸出壓縮到 [-1, 1]
   - 再乘以 2 → 範圍變為 [-2, 2] (符合環境要求)

2. **σ 頭 (標準差)**：
   - 使用 `softplus` 確保 σ > 0 (標準差必須是正數)
   - 加上 1e-5 避免數值不穩定

**輸出：** `tfp.distributions.Normal(loc=μ, scale=σ)`

**程式碼：**
```python
class Actor(nnx.Module):
    def __call__(self, x: jax.Array) -> tfd.Normal:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))

        mu = jnp.tanh(self.fc_mu(x)) * 2.0      # 平均值 [-2, 2]
        sigma = nnx.softplus(self.fc_sigma(x)) + 1e-5  # 標準差 > 0

        return tfd.Normal(loc=mu, scale=sigma)   # 回傳機率分佈
```

#### 2. 評論家 (Critic) - 價值網路 V_φ

**工作：** 評分者 (提供基線)

**網路結構：** 標準 MLP
```
Input (3)  →  FC(64) → ReLU → FC(64) → ReLU → FC_out(1) → V(s)
```

**輸出：** 一個數字，代表「在狀態 s，我預期能獲得的總獎勵」

**程式碼：**
```python
class Critic(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc_out(x)  # 輸出 V(s)
```

### PPO 的三大核心技術

#### 技術 1：Advantage (優勢函數)

**問題：** REINFORCE 使用「絕對總分」作為學習訊號 → 雜訊太高

**解決方案：** 使用「相對分數」

```
Advantage(s, a) = 實際拿到的分數 - Critic 預期的分數
A(s, a) = Q(s, a) - V(s)
```

**訊號解讀：**
- `A > 0`：表現**比預期好** → 增加這個動作的機率 ✅
- `A < 0`：表現**比預期差** → 降低這個動作的機率 ❌
- `A ≈ 0`：表現**符合預期** → 不改變

#### 技術 2：GAE (Generalized Advantage Estimation)

**問題：** 如何準確計算 Advantage？

**方案：** 使用 GAE，這是一種「平滑」的 Advantage 計算方法

**GAE 公式 (從後往前遞迴)：**
```python
for t in reversed(range(N)):
    # 1. 計算 TD 誤差
    delta_t = reward_t + γ × V(s_{t+1}) - V(s_t)

    # 2. 計算 GAE (遞迴)
    A_t = delta_t + γ × λ × A_{t+1}

    # 3. 計算 Return (Critic 的學習目標)
    Return_t = A_t + V(s_t)
```

**超參數：**
- `γ` (GAMMA = 0.99)：折扣因子 (對未來獎勵的重視程度)
- `λ` (GAE_LAMBDA = 0.95)：GAE 的平滑參數
  - `λ = 0`：只看一步 (低變異數，高偏差)
  - `λ = 1`：看到底 (高變異數，低偏差)
  - `λ = 0.95`：折衷方案 ⭐

**最終優化：Advantage 標準化**
```python
advantages = (advantages - mean) / (std + 1e-8)
```
讓 Advantage 的平均值為 0，標準差為 1 → 訓練更穩定

#### 技術 3：PPO-Clip (限制更新步伐)

**問題：** Actor-Critic 訓練不穩定，可能「一步走太大」導致策略崩潰

**解決方案：** PPO-Clip 增加「安全鎖」

**核心概念：策略比例 (Policy Ratio)**
```
Ratio = π_new(a|s) / π_old(a|s)
```
- `Ratio ≈ 1`：新舊策略相似 (安全)
- `Ratio >> 1` 或 `Ratio << 1`：新舊策略差異太大 (危險)

**PPO-Clip Loss 函數：**
```python
# 計算兩種 Loss
loss_unclipped = Advantage × Ratio
loss_clipped = Advantage × clip(Ratio, 1-ε, 1+ε)

# 取較小值 (悲觀原則)
loss = -mean(minimum(loss_unclipped, loss_clipped))
```

**CLIP_EPSILON = 0.2 的含義：**
- Ratio 被限制在 [0.8, 1.2] 範圍內
- 即使 Advantage 很大，策略也**不能**一次更新超過 20%
- 確保訓練穩定

**視覺化：**
```
Advantage > 0 (好動作)
┌────────────────────────────┐
│  允許增加機率，但不超過 20%  │  ← Clip 上限 (1.2)
├────────────────────────────┤
│  正常更新區間 [0.8, 1.2]    │
├────────────────────────────┤
│  允許減少機率，但不超過 20%  │  ← Clip 下限 (0.8)
└────────────────────────────┘

Advantage < 0 (壞動作) - 反過來
```

## PPO 訓練流程

### On-Policy vs Off-Policy

| 特性 | Off-Policy (DQN) | On-Policy (PPO) |
|------|-----------------|----------------|
| **資料來源** | 任何舊策略 | 必須是**當前**策略 |
| **經驗回放** | ✅ Replay Buffer (可重複使用) | ❌ Rollout Buffer (用完即丟) |
| **訓練穩定性** | 較難 (需要 Target Network) | 較易 (策略更新更平滑) |
| **樣本效率** | 高 (一筆資料用多次) | 低 (一筆資料只用一次) |

**為什麼 PPO 是 On-Policy？**
- PPO 的 Loss 計算需要「舊策略的 log 機率」
- 如果資料來自「太舊」的策略，Ratio 會失真
- 因此，PPO 必須在**收集完資料後立刻學習，然後丟棄**

### RolloutBuffer (On-Policy 儲存區)

```python
class RolloutBuffer:
    def add(self, state, action, reward, log_prob, value, done):
        # 儲存一步的經驗

    def calculate_advantages_and_returns(self, last_value, gamma, gae_lambda):
        # 計算 GAE 和 Returns (從後往前)

    def get_data_for_learning(self):
        # 轉換為 JAX 陣列供訓練使用

    def clear(self):
        # 學習完畢後，清空所有資料
```

### PPO 四階段生命週期

```python
while total_steps < MAX_STEPS:
    # ========== 階段 1: 收集 (Rollout) ==========
    for _ in range(ROLLOUT_STEPS):  # 例如 2048 步
        # 1. 選擇動作
        action, value, log_prob = agent.select_action(state)

        # 2. 與環境互動
        next_state, reward, done, _, _ = env.step(action)

        # 3. 儲存到 Buffer
        buffer.add(state, action, reward, log_prob, value, done)

        state = next_state

    # ========== 階段 2: 計算學習目標 (GAE) ==========
    # 取得「最後一步」的 V 值
    last_value = critic(state)

    # 計算所有步驟的 Advantages 和 Returns
    buffer.calculate_advantages_and_returns(last_value, GAMMA, GAE_LAMBDA)

    # ========== 階段 3: 學習 (Learn) ==========
    # 取得所有資料
    states, actions, log_probs_old, advantages, returns = buffer.get_data_for_learning()

    # 反覆訓練 K 次 (TRAIN_EPOCHS = 10)
    for epoch in range(TRAIN_EPOCHS):
        # 打亂資料
        indices = random.permutation(ROLLOUT_STEPS)

        # 分批訓練 (BATCH_SIZE = 64)
        for batch_indices in batches(indices, BATCH_SIZE):
            # 訓練 Critic (最小化 MSE)
            train_critic(batch_states, batch_returns)

            # 訓練 Actor (PPO-Clip Loss)
            train_actor(batch_states, batch_actions,
                       batch_log_probs_old, batch_advantages)

    # ========== 階段 4: 丟棄 (Discard) ==========
    buffer.clear()  # 清空所有「舊策略」的資料
```

## 超參數設定

| 參數 | 值 | 說明 |
|------|-----|------|
| `STATE_DIM` | 3 | 狀態空間維度 (cos θ, sin θ, θ̇) |
| `ACTION_DIM` | 1 | 動作空間維度 (torque) |
| `NUM_TOTAL_TIMESTEPS` | 100,000 | 總訓練步數 |
| `ROLLOUT_STEPS` | 2,048 | 每次收集的步數 (N) |
| `TRAIN_EPOCHS` | 10 | 每批資料訓練的輪數 (K) |
| `BATCH_SIZE` | 64 | Mini-batch 大小 |
| `GAMMA` | 0.99 | 折扣因子 (γ) |
| `GAE_LAMBDA` | 0.95 | GAE 平滑參數 (λ) |
| `CLIP_EPSILON` | 0.2 | PPO 裁剪參數 (ε) |
| `ACTOR_LR` | 3e-4 | Actor 學習率 |
| `CRITIC_LR` | 1e-3 | Critic 學習率 |

**學習率選擇：**
- Critic 的學習率 (1e-3) 比 Actor (3e-4) 高
- 原因：Critic 需要快速學會評估狀態，為 Actor 提供準確的基線

## 預期輸出

### 訓練過程

```
開始 PPO 訓練...

--- 正在收集 2048 步的資料 ---
...正在計算 GAE (Advantages) 和 Returns...
...開始 10 個 Epochs 的學習...
目前總步數: 2048/100000

--- 正在收集 2048 步的資料 ---
...正在計算 GAE (Advantages) 和 Returns...
...開始 10 個 Epochs 的學習...
目前總步數: 4096/100000

--- 正在收集 2048 步的資料 ---
...正在計算 GAE (Advantages) 和 Returns...
...開始 10 個 Epochs 的學習...
目前總步數: 6144/100000

...

目前總步數: 100000/100000
--- 訓練完成！ ---
```

**解讀：**
- 每次收集 2048 步
- 計算 GAE
- 訓練 10 個 Epochs (每個 Epoch 使用所有 2048 筆資料，分成多個 batch)
- 丟棄資料並開始下一輪

### 評估訓練效果

訓練完成後，可以手動評估 Agent 的表現：

```python
# 在 main() 函數最後添加
env = gym.make("Pendulum-v1", render_mode="human")
state, _ = env.reset()

for _ in range(1000):
    action, _, _ = agent.select_action(state)
    state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()
```

**成功的標誌：**
- 擺錘能快速旋轉到正上方
- 在正上方保持穩定 (小幅震盪)
- 平均獎勵 > -200

## 核心程式碼解析

### 1. Actor 的 select_action (JAX ↔ NumPy 橋樑)

```python
def select_action(self, state: np.ndarray):
    # NumPy → JAX (增加 batch 維度)
    state_jnp = jnp.asarray(state[np.newaxis, :], dtype=jnp.float32)

    # 呼叫 Actor → 取得機率分佈
    action_dist = self.actor(state_jnp)  # N(μ, σ)

    # 呼叫 Critic → 取得基線
    value = self.critic(state_jnp)  # V(s)
    value = jax.lax.stop_gradient(value)  # 阻止梯度回傳

    # 從分佈中採樣動作
    action = action_dist.sample(seed=rng_key)

    # 計算 log 機率 (PPO 必須)
    log_prob = action_dist.log_prob(action)

    # JAX → NumPy (移除 batch 維度)
    return action.flatten(), value.flatten(), log_prob.flatten()
```

**為什麼需要 log_prob？**
- PPO 需要計算 `Ratio = exp(log_prob_new - log_prob_old)`
- 必須在「採樣當下」記錄 log_prob_old

### 2. Critic 訓練 (MSE Loss)

```python
def critic_loss_fn(critic_model: Critic):
    values_pred = critic_model(batch_states)  # 預測的 V(s)
    loss = jnp.mean((batch_returns - values_pred.flatten()) ** 2)
    return loss

# 計算梯度並更新
_, critic_grads = nnx.value_and_grad(critic_loss_fn)(self.critic)
self.critic_optimizer.update(critic_grads)
```

**目標：** 讓 V(s) 盡可能接近「實際總分」(Returns)

### 3. Actor 訓練 (PPO-Clip Loss)

```python
def actor_loss_fn(actor_model: Actor):
    # 1. 取得新的 log 機率
    action_dist_new = actor_model(batch_states)
    log_probs_new = action_dist_new.log_prob(batch_actions)

    # 2. 計算 Ratio
    ratio = jnp.exp(log_probs_new - batch_log_probs_old)

    # 3. 計算未裁剪的 Loss
    loss_unclipped = batch_advantages * ratio

    # 4. 計算裁剪的 Loss
    ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    loss_clipped = batch_advantages * ratio_clipped

    # 5. 取最小值 (悲觀原則)
    loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))
    return loss

# 計算梯度並更新
_, actor_grads = nnx.value_and_grad(actor_loss_fn)(self.actor)
self.actor_optimizer.update(actor_grads)
```

**關鍵：** 加負號 `-` 是因為 Adam 只能「最小化」，而我們要「最大化」Advantage

## Q-Learning → DQN → PPO 演進總結

| 特性 | Q-Learning | DQN | PPO |
|------|-----------|-----|-----|
| **學習對象** | Q 值 (價值) | Q 值 (價值) | 策略 (Policy) |
| **函數近似** | ❌ Q-Table | ✅ 神經網路 | ✅ 神經網路 (Actor + Critic) |
| **動作空間** | 離散 | 離散 | **連續 + 離散** |
| **策略類型** | Off-Policy | Off-Policy | **On-Policy** |
| **經驗回放** | ❌ | ✅ Replay Buffer | ❌ (Rollout Buffer) |
| **穩定技術** | ❌ | Target Network | **PPO-Clip + GAE** |
| **優勢函數** | ❌ | ❌ | ✅ |
| **適用場景** | 小狀態空間 | 大狀態空間 + 離散動作 | **任何場景** (最通用) |

## PPO 的實際應用

PPO 是目前工業界最受歡迎的 RL 演算法，應用包括：

1. **機器人控制**
   - 機械臂抓取
   - 四足機器人行走
   - 無人機飛行

2. **遊戲 AI**
   - OpenAI Five (Dota 2)
   - AlphaStar (StarCraft II)
   - 各種連續控制遊戲

3. **大型語言模型對齊 (RLHF)**
   - ChatGPT 的訓練
   - Claude 的訓練
   - **GRPO** (Group Relative Policy Optimization) 是 PPO 的變體

4. **自動駕駛**
   - 路徑規劃
   - 速度控制

## 進階主題

### PPO 的變體

1. **PPO-Penalty**
   - 使用 KL 散度懲罰代替 Clip
   - `Loss = Advantage - β × KL(π_new || π_old)`

2. **GRPO** (用於 LLM)
   - Group Relative Policy Optimization
   - 專為大型語言模型設計的 PPO 變體

### 進一步優化

1. **Vectorized Environments**
   - 同時運行多個環境副本
   - 加速資料收集

2. **Normalization**
   - 狀態標準化
   - 獎勵標準化

3. **Learning Rate Scheduling**
   - 學習率遞減
   - 提高訓練後期的穩定性

## Flax NNX 關鍵 API 總結

### 1. 多網路管理

```python
# 建立兩個獨立的網路
actor_key, critic_key = jax.random.split(rng_key)
self.actor = Actor(..., rngs=nnx.Rngs(actor_key))
self.critic = Critic(..., rngs=nnx.Rngs(critic_key))
```

### 2. 多優化器管理

```python
# 每個網路有自己的優化器
self.actor_optimizer = nnx.Optimizer(self.actor, optax.adam(3e-4))
self.critic_optimizer = nnx.Optimizer(self.critic, optax.adam(1e-3))
```

### 3. 梯度阻斷

```python
# 在 select_action 時阻止 Critic 的梯度
value = self.critic(state_jnp)
value = jax.lax.stop_gradient(value)  # 不訓練 Critic
```

### 4. RNG 流管理

```python
# 建立 RNG 流
self.rng_stream = nnx.Rngs(jax.random.PRNGKey(42))

# 在需要隨機性時取得新密鑰
rng_key = self.rng_stream.sampler()
action = action_dist.sample(seed=rng_key)
```

## 參考資料

- Schulman et al. (2017). "Proximal Policy Optimization Algorithms" ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
- Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation" ([arXiv:1506.02438](https://arxiv.org/abs/1506.02438))
- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 13: Policy Gradient Methods)
- [Gymnasium Pendulum-v1 Documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
