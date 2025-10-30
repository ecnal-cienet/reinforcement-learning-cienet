# Group Relative Policy Optimization (GRPO) 實作 - MNIST 分類

## 概述

這是一個將 **Group Relative Policy Optimization (GRPO)** 演算法應用於 **MNIST 手寫數字分類** 的實作。GRPO 是 PPO 的變體，專為大型語言模型 (LLM) 訓練而設計，其核心創新是**移除 Critic 網路**以節省記憶體。

**核心突破：** GRPO 使用「群組平均表現」作為基線，取代了 PPO 中需要額外訓練的 Critic 網路，從而大幅降低記憶體需求。

**與前三個專案的關係：**
- `1_Q_Learning/`: 價值為基礎 - 表格型
- `2_Cart_Pole_DQN/`: 價值為基礎 - 深度學習
- `3_Pendulum/`: 策略為基礎 - Actor-Critic (PPO)
- `4_GRPO_MNIST/` **(本專案)**: 策略為基礎 - **Critic-less** (GRPO)

## 為什麼需要 GRPO？

### PPO 在大型模型訓練中的記憶體瓶頸

當訓練像 ChatGPT 這樣的大型語言模型 (例如 8B 參數) 時，PPO 面臨嚴重的記憶體問題：

**PPO 需要：**
1. **Actor 網路** (80 億參數)
   - 權重: 16 GB
   - 梯度: 16 GB
   - Adam 狀態: 32 GB
   - **小計**: 64 GB

2. **Critic 網路** (80 億參數)
   - 權重: 16 GB
   - 梯度: 16 GB
   - Adam 狀態: 32 GB
   - **小計**: 64 GB

**總需求**: 128 GB (僅模型訓練狀態)

這在單一晶片 (通常 32-80 GB HBM) 上是不可行的，即使使用分散式訓練也成本高昂。

### GRPO 的解決方案

**核心思想：** 完全移除 Critic 網路，使用「**群組相對基線 (Group Relative Baseline)**」

**GRPO 的基線計算：**
```python
# PPO: 需要訓練一個 Critic 網路來預測 V(s)
baseline = Critic(state)  # ← 80 億參數的網路！

# GRPO: 動態計算群組平均
baseline = mean(group_rewards)  # ← 只需要一個 mean() 操作！
```

**優勢：**
- ✅ **記憶體節省**: 省下整個 Critic 網路 (64 GB)
- ✅ **訓練簡化**: 不需要訓練和維護 Critic
- ✅ **無 Critic 訓練不穩定**: 避免 Critic 誤導 Actor 的問題

**代價：**
- ❌ **收集成本更高**: 需要為每個狀態生成 `G` 次回答來計算群組平均
- ❌ **樣本效率較低**: 相同狀態需要多次推理

**結論：** GRPO 是一種**犧牲推理成本 (資料收集) 換取記憶體效率 (訓練) 的演算法**，特別適合 LLM 這種單次推理成本相對較低，但模型訓練記憶體需求極高的場景。

## 環境說明

### MNIST 作為 RL 問題

我們將經典的監督式學習問題 (MNIST 分類) 重新框架為強化學習問題：

**傳統監督式學習：**
```
輸入圖片 → 模型預測 → 計算 Loss (Cross-Entropy) → 更新權重
```

**RL 框架 (GRPO)：**
```
State (圖片) → Actor 採樣動作 (預測數字) → 獲得 Reward (對/錯) → 計算 Advantage → 更新 Actor
```

### RL 環境參數

- **狀態空間 (State Space)**：連續 784 維向量
  - 28×28 灰階圖片扁平化為 (784,)
  - 值域：0.0 ~ 1.0 (正規化後)

- **動作空間 (Action Space)**：離散 10 個動作
  - 動作 0-9 代表預測的數字
  - 使用 `tfp.distributions.Categorical` 來表示動作機率分佈

- **獎勵函數**：
  ```python
  reward = 1.0  # 預測正確
  reward = 0.0  # 預測錯誤
  ```
  - 這是一個「稀疏獎勵 (Sparse Reward)」問題
  - 只有完全答對才有獎勵

- **回合結構**：
  - **單步回合 (One-Step Episode)**：每張圖片是一個獨立的回合
  - 猜完數字後回合立即結束
  - 不需要考慮時序依賴性

- **成功標準**：
  - MNIST 訓練集準確率 > 95% (良好)
  - MNIST 訓練集準確率 > 97% (優秀)

## 執行方式

### 前置條件

確保已啟動虛擬環境並安裝相依套件：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**特別注意：** 本專案需要以下額外依賴：
- `tensorflow-datasets`: 載入 MNIST 資料集
- `tensorflow-probability[jax]`: 提供 `Categorical` 分佈

### 執行程式

```bash
python 4_GRPO_MNIST/grpo_mnist.py
```

或者在 `4_GRPO_MNIST` 目錄內執行：

```bash
cd 4_GRPO_MNIST
python grpo_mnist.py
```

## 演算法核心

### GRPO vs PPO 對比

| 特性 | PPO | GRPO |
|------|-----|------|
| **網路結構** | Actor + Critic | **只有 Actor** |
| **基線 (Baseline)** | `V(s)` (Critic 網路預測) | `mean(group_rewards)` (動態計算) |
| **優勢計算** | `A = R - V(s)` | `A = R - mean(R_group)` |
| **記憶體需求** | 高 (兩個網路) | **低 (一個網路)** |
| **收集成本** | 1× (每個 s 生成 1 次) | **G× (每個 s 生成 G 次)** |
| **適用場景** | 通用 RL | **大型模型訓練 (LLM)** |

### GRPO 的「群組」概念

**核心問題：** 在沒有 Critic 的情況下，如何判斷一個 Reward 是「好」還是「壞」？

**GRPO 的答案：** 使用「群組相對比較」

**範例：**
```
假設我們有一批 (G=1024) 張圖片：
- 圖片 1: Agent 猜「7」，正確 → R₁ = 1.0
- 圖片 2: Agent 猜「3」，錯誤 → R₂ = 0.0
- 圖片 3: Agent 猜「5」，正確 → R₃ = 1.0
- ...
- 圖片 1024: Agent 猜「2」，錯誤 → R₁₀₂₄ = 0.0

群組平均: baseline = mean([R₁, R₂, ..., R₁₀₂₄]) = 0.78

相對優勢:
- 圖片 1: A₁ = 1.0 - 0.78 = +0.22 (好！比平均好 22%)
- 圖片 2: A₂ = 0.0 - 0.78 = -0.78 (差！比平均差 78%)
- 圖片 3: A₃ = 1.0 - 0.78 = +0.22 (好！)
```

**關鍵洞察：**
- 即使「答對」(R=1.0)，如果群組平均已經很高 (例如 0.95)，Advantage 也只有 +0.05 (更新幅度小)
- 即使「答錯」(R=0.0)，如果群組平均也很低 (例如 0.20)，Advantage 只有 -0.20 (懲罰較輕)
- 這種「相對比較」讓 Actor 學習的是「在當前能力水平下的相對表現」

### GRPO 的四階段訓練流程

```python
for epoch in range(NUM_EPOCHS):
    for batch in mnist_dataset:  # 每個 batch 就是一個「群組」

        # ========== 階段 1: 收集 (Rollout) ==========
        # 為群組中的每張圖片採樣動作
        actions, log_probs_old = actor.select_actions_and_log_probs(images)

        # ========== 階段 2: 計算相對優勢 (GRPO 核心) ==========
        # (A) 計算獎勵
        rewards = (actions == labels).astype(float)  # 對或錯

        # (B) 計算群組基線 (取代 Critic)
        baseline = mean(rewards)

        # (C) 計算相對優勢
        advantages = rewards - baseline

        # (D) 標準化 (和 PPO 一樣)
        advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)

        # ========== 階段 3: 學習 (PPO-Clip Loss) ==========
        # 使用和 PPO 相同的 Clip Loss 來訓練 Actor
        train_actor(images, actions, log_probs_old, advantages)
```

**關鍵差異：**
1. ❌ **沒有 RolloutBuffer**: 因為是單步問題，不需要儲存多步經驗
2. ❌ **沒有 GAE 計算**: 因為沒有時序依賴性，直接用 `R - baseline`
3. ❌ **沒有 Critic 訓練**: GRPO 的核心特色
4. ✅ **保留 PPO-Clip**: 確保策略更新穩定

## 網路架構

### Actor (唯一的神經網路)

使用 **Flax NNX** 實作的 3 層全連接神經網路 (MLP)：

```
Input (784)  →  FC(128)  →  ReLU  →  FC(128)  →  ReLU  →  FC(10)  →  Categorical
  圖片向量        隱藏層1              隱藏層2              Logits      機率分佈
```

**輸出：** `tfp.distributions.Categorical(logits)`
- 10 個 logits (原始分數) 會被自動轉換為機率分佈
- 例如：`[0.1, 0.05, 0.3, 0.02, ...]` (10 個數字的機率)

**程式碼：**
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
        return tfd.Categorical(logits=logits)  # 回傳離散機率分佈
```

**範例：**
```python
image = [0.1, 0.2, ..., 0.05]  # 784 維向量
dist = actor(image)             # Categorical 分佈
action = dist.sample()          # 採樣 → 可能得到 7
log_prob = dist.log_prob(action)  # log P(動作=7|狀態)
```

## 超參數設定

| 參數 | 值 | 說明 |
|------|-----|------|
| `STATE_DIM` | 784 | 狀態空間維度 (28×28) |
| `ACTION_DIM` | 10 | 動作空間維度 (0-9) |
| `NUM_EPOCHS` | 10 | 訓練輪數 |
| `BATCH_SIZE` | 1,024 | **群組大小 (G)** |
| `LEARNING_RATE` | 1e-4 | Actor 學習率 |
| `CLIP_EPSILON` | 0.2 | PPO 裁剪參數 (ε) |

**BATCH_SIZE 的雙重含義：**
1. **傳統意義**：每批次處理的資料量 (Mini-batch)
2. **GRPO 意義**：**群組大小 (G)**，用於計算相對基線

**為什麼 BATCH_SIZE 大 (1024) 比較好？**
- 群組越大，`mean(rewards)` 越穩定
- 基線越準確，Advantage 的訊號越乾淨
- 但也增加了記憶體和計算成本

## 核心程式碼解析

### 1. 動作選擇與 Log 機率計算

```python
def select_actions_and_log_probs(self, batch_states: jax.Array):
    # 1. 取得機率分佈
    action_dist = self.actor(batch_states)  # Categorical(logits=[...])

    # 2. 採樣動作 (為 G=1024 張圖片各採樣一個數字)
    rng_key = self.rng_stream.sampler()
    actions = action_dist.sample(seed=rng_key)

    # 3. 計算「舊」Log 機率 (PPO 必須)
    log_probs = action_dist.log_prob(actions)

    return actions, log_probs
```

**為什麼需要 log_prob？**
- PPO-Clip 需要計算 `ratio = π_new / π_old`
- 在實作中使用 `ratio = exp(log_prob_new - log_prob_old)` 避免數值不穩定

### 2. GRPO 相對優勢計算

```python
# (A) 計算獎勵 (對或錯)
@jax.jit
def calculate_rewards(actions, labels):
    return jnp.where(actions == labels, 1.0, 0.0)

batch_rewards = calculate_rewards(batch_actions, batch_labels)

# (B) 計算群組基線 (GRPO 核心！)
baseline = jnp.mean(batch_rewards)  # ← 取代了 Critic(state)

# (C) 計算相對優勢
batch_advantages = batch_rewards - baseline

# (D) 標準化 (和 PPO 一樣)
adv_mean = jnp.mean(batch_advantages)
adv_std = jnp.std(batch_advantages) + 1e-8
batch_advantages = (batch_advantages - adv_mean) / adv_std
```

**關鍵洞察：**
```python
# PPO 需要額外訓練 Critic
baseline = critic(state)  # 80 億參數

# GRPO 只需要一個 mean 操作
baseline = mean(rewards)  # 零參數！
```

### 3. Actor 訓練 (PPO-Clip Loss)

```python
def actor_loss_fn(actor_model: Actor):
    # (1) 取得新的機率分佈
    action_dist_new = actor_model(batch_states)
    log_probs_new = action_dist_new.log_prob(batch_actions)

    # (2) 計算策略比例
    ratio = jnp.exp(log_probs_new - batch_log_probs_old)

    # (3) 計算未裁剪 Loss
    loss_unclipped = batch_advantages * ratio

    # (4) 計算裁剪 Loss (PPO 核心)
    ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    loss_clipped = batch_advantages * ratio_clipped

    # (5) 取較小值 (悲觀原則)
    loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))
    return loss

# 計算梯度並更新
_, actor_grads = nnx.value_and_grad(actor_loss_fn)(self.actor)
self.actor_optimizer.update(actor_grads)
```

**這和 PPO 的 Actor 訓練完全相同！**
- 唯一的差異是 Advantage 的計算方式 (群組相對 vs Critic 預測)

## 預期輸出

### 訓練過程

```
開始 GRPO on MNIST 訓練...
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
--- 訓練完成！ ---
```

**解讀：**
- **初期快速提升** (Epoch 1-3)：Agent 學會基本的數字識別
- **中期穩定成長** (Epoch 4-6)：策略持續優化
- **後期收斂** (Epoch 7-10)：準確率接近極限
- **成功標準**：準確率 > 95% 表示 GRPO 成功應用於監督式學習問題

### 效能分析

**記憶體節省：**
- PPO: 需要 Actor (128→128→10) + Critic (128→128→1)
- GRPO: 只需要 Actor (128→128→10)
- **節省約 50% 的網路參數** (在 LLM 場景下節省更顯著)

**計算成本：**
- PPO: 每個樣本採樣 1 次
- GRPO: 每個批次需要 G 個樣本來計算群組基線
- **推理成本增加約 G 倍** (但在 LLM 場景下，推理成本 << 訓練記憶體成本)

## Q-Learning → DQN → PPO → GRPO 演進總結

| 特性 | Q-Learning | DQN | PPO | GRPO |
|------|-----------|-----|-----|------|
| **學習對象** | Q 值 | Q 值 | 策略 | 策略 |
| **函數近似** | Q-Table | 神經網路 | 神經網路 (A+C) | **神經網路 (A)** |
| **動作空間** | 離散 | 離散 | 連續+離散 | 連續+離散 |
| **基線方法** | ❌ 無 | ❌ 無 | Critic 網路 | **群組平均** |
| **網路數量** | 0 | 2 (Online+Target) | 2 (Actor+Critic) | **1 (Actor)** |
| **記憶體需求** | 低 | 中 | 高 | **低** |
| **樣本效率** | 低 | 高 (Replay) | 低 (On-Policy) | **最低 (G×)** |
| **適用場景** | 小狀態空間 | 大狀態空間 | 通用 RL | **大型模型訓練** |

## GRPO 在 LLM 訓練中的應用

### 將 MNIST 類比到 LLM

| MNIST (本專案) | LLM (ChatGPT/Claude) |
|---------------|---------------------|
| 狀態: 圖片 (784 維) | 狀態: Prompt + 歷史 (上下文) |
| 動作: 預測數字 (0-9) | 動作: 生成下一個 Token |
| 獎勵: 正確/錯誤 (0/1) | 獎勵: 人類偏好分數 (Reward Model) |
| 群組大小: BATCH_SIZE=1024 | 群組大小: 每個 Prompt 生成 G 個回答 |
| 基線: `mean(群組答對率)` | 基線: `mean(群組獎勵分數)` |

### LLM 中的 GRPO 流程

```python
# 假設訓練 ChatGPT
for prompt in training_prompts:
    # ========== 階段 1: 收集 (多次採樣) ==========
    # 為同一個 Prompt 生成 G=8 個不同回答
    responses = []
    for _ in range(G=8):
        response = actor.generate(prompt)  # 從 LLM 採樣
        responses.append(response)

    # ========== 階段 2: 計算相對優勢 ==========
    # 使用 Reward Model 評分
    rewards = [reward_model(prompt, resp) for resp in responses]
    # rewards = [7.2, 8.5, 6.1, 9.0, 7.8, 6.5, 8.2, 7.5]

    # 計算群組基線
    baseline = mean(rewards)  # = 7.6

    # 計算相對優勢
    advantages = rewards - baseline
    # advantages = [-0.4, +0.9, -1.5, +1.4, +0.2, -1.1, +0.6, -0.1]

    # ========== 階段 3: 學習 (PPO-Clip) ==========
    # 訓練 LLM 提高「好」回答的機率，降低「差」回答的機率
    train_actor(prompt, responses, advantages)
```

**為什麼 LLM 訓練特別適合 GRPO？**
1. **記憶體瓶頸嚴重**: 80 億參數的 Critic 成本太高
2. **推理成本相對可控**: 生成 8 次回答的成本 << 訓練 80 億參數 Critic 的成本
3. **群組比較合理**: 同一個 Prompt 的多個回答天然形成「群組」

## 實作限制與延伸

### 本專案的簡化

1. **單步問題**
   - MNIST 不需要時序推理
   - 真實 LLM 是多步序列生成問題 (需要處理每個 Token 的 Advantage)

2. **稀疏獎勵**
   - 只有 0/1 兩種獎勵
   - 真實 LLM 使用 Reward Model 提供連續分數

3. **無 Reference Model**
   - 本實作未使用 KL 散度懲罰
   - 真實 GRPO 通常搭配 `BETA × KL(π_new || π_ref)` 來進一步穩定訓練

### 進階優化

1. **KL 散度約束**
   ```python
   # 防止策略偏離「基礎模型」太遠
   kl_penalty = beta × KL(actor || reference_model)
   loss = -advantages + kl_penalty
   ```

2. **自適應 Baseline**
   ```python
   # 使用移動平均作為基線
   baseline = 0.9 × baseline + 0.1 × mean(rewards)
   ```

3. **多輪優化**
   ```python
   # 和 PPO 一樣，對同一批資料訓練多個 Epochs
   for epoch in range(K_EPOCHS):
       train_actor(...)
   ```

## 參考資料

- DeepSeek-R1 Technical Report (2025). "Group Relative Policy Optimization for RLHF"
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms" ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
- OpenAI. "Learning to Summarize from Human Feedback" (介紹 RLHF 流程)
- Anthropic. "Training a Helpful and Harmless Assistant with RLHF" (Claude 訓練方法)

## 總結

GRPO 是 PPO 針對大型模型訓練場景的一種聰明變體：

**核心取捨：**
- ✅ **犧牲**: 推理成本 (需要 G 次採樣)
- ✅ **換取**: 記憶體效率 (移除 Critic 網路)

**適用場景：**
- ✅ 大型語言模型訓練 (ChatGPT, Claude, Llama)
- ✅ 記憶體受限的場景
- ❌ 小模型 (PPO 更高效)
- ❌ 推理成本極高的場景

這個 MNIST 專案展示了 GRPO 的核心概念，雖然是一個簡化的單步問題，但其「群組相對基線」的思想完全適用於 LLM 的多步序列生成場景。
