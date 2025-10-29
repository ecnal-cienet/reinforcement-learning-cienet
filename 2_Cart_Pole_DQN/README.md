# Deep Q-Network (DQN) 實作 - CartPole-v1

## 概述

這是一個使用 **Deep Q-Network (DQN)** 演算法來解決 OpenAI Gymnasium 的 **CartPole-v1** 環境的實作。DQN 是第一個成功結合深度學習與強化學習的演算法，由 DeepMind 於 2015 年提出，能夠處理高維度狀態空間的問題。

**核心突破：** DQN 使用**深度神經網路**作為函數近似器 (Function Approximator) 來取代傳統的 Q-Table，從而解決了表格型方法的「維度詛咒」問題。

## 環境說明

### CartPole-v1 (倒立擺)

CartPole 是經典的控制問題：一根桿子通過非驅動關節附著在小車上，目標是通過左右移動小車來保持桿子直立。

```
        |
        |  ← 桿子 (Pole)
        |
    ┌───────┐
    │ 小車  │ ← 可左右移動
    └───┬───┘
    ════════════
```

### 環境參數

- **狀態空間 (State Space)**：連續 4 維向量
  - `position`: 小車位置 (範圍: -4.8 ~ 4.8)
  - `velocity`: 小車速度
  - `angle`: 桿子角度 (範圍: -0.418 ~ 0.418 弧度，約 ±24°)
  - `angular_velocity`: 桿子角速度

- **動作空間 (Action Space)**：離散 2 個動作
  - `0`: 向左推小車
  - `1`: 向右推小車

- **獎勵函數**：
  - 每存活一個時間步獲得 `+1` 分
  - 目標是盡可能長時間保持桿子直立

- **終止條件**：
  - 桿子傾斜角度超過 ±12°
  - 小車移出邊界
  - 達到最大步數 (500 步)

- **成功標準**：
  - 連續 100 個回合的平均獎勵 ≥ 475

## 執行方式

### 前置條件

確保已啟動虛擬環境並安裝相依套件：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 執行程式

```bash
python 2_Cart_Pole_DQN/cart_pole_dqn.py
```

或者在 `2_Cart_Pole_DQN` 目錄內執行：

```bash
cd 2_Cart_Pole_DQN
python cart_pole_dqn.py
```

## 演算法核心

### 為什麼需要 DQN？

**Q-Learning 的問題：**
- CartPole 的狀態空間是**連續**的 (例如：位置 = 1.234567...)
- 無法為「無限」多個狀態建立 Q-Table

**DQN 的解決方案：**
- 使用**神經網路** $Q_\theta(s, a)$ 來**估算** Q 值
- 網路可以「泛化」：相似的狀態會產生相似的 Q 值

### DQN 的兩大穩定技術

直接用神經網路訓練 Q-Learning 會非常不穩定。DQN 引入了兩項關鍵技術：

#### 1. 經驗回放 (Experience Replay)

**問題：** 神經網路訓練最怕「高度相關」的連續資料，會導致過度擬合 (Overfitting)。

**解決方案：** 建立一個 **Replay Buffer** (記憶體緩衝區)

```python
class ReplayBuffer:
    - 容量 (BUFFER_SIZE): 10,000 筆經驗
    - 儲存格式: (state, action, reward, next_state, done)
    - 訓練時隨機抽樣 BATCH_SIZE=64 筆不相關的經驗
```

**好處：**
- ✅ 打破資料的時間相關性
- ✅ 重複利用過去的經驗 (資料效率高)
- ✅ 訓練更穩定

#### 2. 目標網路 (Target Network)

**問題：** 「移動的靶心」問題

在傳統 Q-Learning 中，我們同時用**同一個網路**來計算「預測值」和「目標值」：

```
Loss = [R + γ × max Q(S', a') - Q(S, A)]²
         └──────┬──────┘   └───┬───┘
              目標值        預測值
           (都來自同一個網路)
```

這就像你在射擊一個**你自己控制**的靶心 → 永遠追不上！

**解決方案：** 使用**兩個**神經網路

1. **線上網路 (Online Network)** $Q_{\text{online}}$
   - 負責：選擇動作、計算預測值
   - 狀態：**每步都更新**

2. **目標網路 (Target Network)** $Q_{\text{target}}$
   - 負責：計算 TD 目標值
   - 狀態：**權重被凍結** (每 100 步才同步一次)

```python
# 計算 TD 目標 (使用 Target Network - 固定的靶心)
q_next_target = self.target_network(next_states)
td_target = rewards + GAMMA * jnp.max(q_next_target, axis=1)

# 計算 Loss (使用 Online Network)
q_current = self.online_network(states)
loss = mean((q_current - td_target)²)
```

**同步機制：**
```python
if total_steps % TARGET_UPDATE_FREQ == 0:  # 每 100 步
    agent.update_target_network()  # 複製權重
```

## 網路架構

### QNetwork (函數近似器)

使用 **Flax NNX** 框架實作的 3 層全連接神經網路 (MLP)：

```
Input (4)  →  FC (64)  →  ReLU  →  FC (64)  →  ReLU  →  FC (2)  →  Output
 狀態向量      隱藏層1             隱藏層2              Q值 (左, 右)
```

**實作程式碼：**
```python
class QNetwork(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 64, rngs=rngs)
        self.fc3 = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)  # 輸出 Q 值 (Logits)
```

**輸入範例：**
```python
state = [0.02, 0.01, -0.03, 0.04]  # [position, velocity, angle, angular_velocity]
q_values = network(state)  # 輸出: [Q(s, 左), Q(s, 右)] = [1.23, 2.45]
action = argmax(q_values)  # 選擇 Q 值最高的動作 → 右 (1)
```

## 超參數設定

| 參數 | 值 | 說明 |
|------|-----|------|
| `STATE_DIM` | 4 | 狀態空間維度 |
| `ACTION_DIM` | 2 | 動作空間維度 |
| `BUFFER_SIZE` | 10,000 | Replay Buffer 容量 |
| `BATCH_SIZE` | 64 | 訓練批次大小 |
| `GAMMA` | 0.99 | 折扣因子 |
| `LEARNING_RATE` | 0.001 | Adam 優化器學習率 |
| `NUM_EPISODES` | 500 | 訓練回合數 |
| `TARGET_UPDATE_FREQ` | 100 | 目標網路更新頻率 (步數) |
| `EPSILON_START` | 1.0 | 初始探索率 |
| `EPSILON_END` | 0.01 | 最終探索率 |
| `EPSILON_DECAY` | 0.995 | Epsilon 衰退率 (每回合) |

### Epsilon 衰退策略

使用**指數衰退 (Exponential Decay)**：

```python
epsilon = max(EPSILON_END, epsilon × EPSILON_DECAY)
```

這與 Q-Learning 的**線性衰退**不同，能更快速地從「探索」轉向「利用」。

## DQN Agent 核心流程

### 1. 初始化

```python
agent = DQNAgent(STATE_DIM, ACTION_DIM, rng_key=rng_key)
```

建立：
- Online Network (可訓練)
- Target Network (權重凍結)
- Replay Buffer (經驗池)
- Optimizer (Adam)

### 2. 動作選擇 (Epsilon-Greedy)

```python
def select_action(self, state, rng_key):
    if random() <= epsilon:
        return random_action()  # 探索
    else:
        q_values = self.online_network(state)
        return argmax(q_values)  # 利用
```

### 3. 訓練步驟

```python
def train_step(self):
    # 1. 從 Replay Buffer 中抽樣
    states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

    # 2. 計算 TD 目標 (使用 Target Network)
    q_next_target = self.target_network(next_states)
    td_target = rewards + GAMMA * max(q_next_target) * (1 - dones)

    # 3. 定義 Loss 函數
    def loss_fn(model):
        q_current = model(states)
        q_current_action = q_current[actions]
        return mean((q_current_action - td_target)²)

    # 4. 計算梯度並更新 Online Network
    _, grads = nnx.value_and_grad(loss_fn)(self.online_network)
    self.optimizer.update(grads)
```

### 4. 主訓練迴圈

```python
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    while not done:
        # (1) 選擇動作
        action = agent.select_action(state, rng_key)

        # (2) 執行動作
        next_state, reward, done, _, _ = env.step(action)

        # (3) 儲存經驗到 Replay Buffer
        agent.buffer.add(state, action, reward, next_state, done)

        # (4) 訓練 Online Network
        agent.train_step()

        # (5) 定期更新 Target Network
        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        state = next_state
        episode_reward += reward

    # (6) Epsilon 衰退
    agent.update_epsilon()
```

## 預期輸出

### 訓練過程

程式會每 50 個回合輸出一次訓練進度：

```
開始訓練 DQN Agent...
Episode 50, Epsilon: 0.779, Avg Reward (last 50): 22.34
Episode 100, Epsilon: 0.606, Avg Reward (last 50): 45.12
Episode 150, Epsilon: 0.472, Avg Reward (last 50): 98.56
...同步 Target Network 權重...
Episode 200, Epsilon: 0.368, Avg Reward (last 50): 165.78
Episode 250, Epsilon: 0.286, Avg Reward (last 50): 234.12
...同步 Target Network 權重...
Episode 300, Epsilon: 0.223, Avg Reward (last 50): 312.45
Episode 350, Epsilon: 0.174, Avg Reward (last 50): 421.67
Episode 400, Epsilon: 0.135, Avg Reward (last 50): 487.23  ← 成功！
訓練完成！
```

**解讀：**
- **Epsilon** 持續下降：從探索轉向利用
- **Avg Reward** 逐漸上升：Agent 越來越聰明
- **目標網路同步**：每隔一段時間會看到同步訊息
- **成功標準**：平均獎勵 ≥ 475

## Flax NNX 關鍵 API

本實作使用 **Flax NNX** (新一代 API) 而非舊版的 `flax.linen`：

### 1. 模型定義

```python
class QNetwork(nnx.Module):
    def __init__(self, ..., *, rngs: nnx.Rngs):  # 必須接收 rngs
        self.fc1 = nnx.Linear(...)
```

### 2. 優化器綁定

```python
self.optimizer = nnx.Optimizer(self.online_network, optax.adam(LEARNING_RATE))
```

### 3. 權重提取與更新 (Target Network 同步)

```python
# 提取 Online Network 的權重
online_state = nnx.state(self.online_network)

# 更新 Target Network 的權重
nnx.update(self.target_network, online_state)
```

### 4. 梯度計算與更新

```python
# 計算 Loss 和梯度
_, grads = nnx.value_and_grad(loss_fn)(self.online_network)

# 使用優化器更新參數
self.optimizer.update(grads)
```

## Q-Learning vs DQN 對比

| 特性 | Q-Learning (表格型) | DQN (深度學習) |
|------|-------------------|---------------|
| **Q 值儲存** | Q-Table (NumPy 陣列) | 神經網路 |
| **狀態空間** | 小規模離散狀態 (16 個) | 高維度連續狀態 (無限) |
| **記憶體需求** | O(狀態數 × 動作數) | O(網路參數數) |
| **泛化能力** | ❌ 無 (每個狀態獨立) | ✅ 有 (相似狀態共享知識) |
| **經驗回放** | ❌ 不使用 | ✅ Replay Buffer |
| **目標網路** | ❌ 不需要 | ✅ Target Network |
| **更新方式** | 直接更新 Q(s,a) | 梯度下降 |
| **Epsilon 衰退** | 線性衰退 | 指數衰退 |

## 進階技術與延伸

本實作使用的是基礎 DQN。以下是後續的改進版本：

1. **Double DQN** (DDQN)
   - 解決 Q 值過估計問題
   - 使用 Online Network 選擇動作，Target Network 評估價值

2. **Dueling DQN**
   - 網路分為兩個分支：V(s) 和 A(s,a)
   - 更好的價值估計

3. **Prioritized Experience Replay** (PER)
   - 優先回放「重要」的經驗
   - 加速學習

4. **Rainbow DQN**
   - 結合上述所有技術的最強版本

## 從 DQN 到 Policy Gradient

**DQN 的局限：**
- ❌ 無法處理**連續動作空間** (例如：方向盤角度 -180° ~ 180°)
- ❌ 只能「間接」學習策略 (通過 Q 值)

**下一步：Policy Gradient 方法**
- 參見 `3_Pendulum/` 的 PPO 實作
- 直接學習策略 π(a|s)
- 可處理連續動作

## 參考資料

- Mnih et al. (2015). "Human-level control through deep reinforcement learning" (Nature)
- Mnih et al. (2013). "Playing Atari with Deep Reinforcement Learning" (NIPS Workshop)
- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 9-11)
- [Gymnasium CartPole-v1 Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
