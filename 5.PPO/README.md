## <a name="module-5-actor-critic--ppo"></a>模組五：Actor-Critic 與 PPO (PPO/GRPO 的核心)

[⬅️ 上一章：模組四 - Policy Gradients](../4.Policy_Gradients/README.md) | [返回目錄](../README.md) | [下一章：模組六 - JAX/Maxtext ➡️](../6.Jax&Maxtext/README.md)

---

本模組是現代 RL 的核心，它**修正了** REINFORCE 的所有缺陷，並直達你工作中的 `PPO/GRPO`。

### 5.1 基線 (Baseline) 與 優勢 (Advantage)

為了修正 REINFORCE，我們需要一個「**相對分數**」，而不是「絕對分數」。

**1. 引入「基線 (Baseline)」**
* **理論**：我們需要一個「標準」來判斷 `+200` 分是好是壞。
* **最好的基線**：就是「**價值函數 $V(s)$**」 (來自模組一)。
* $V(s)$ 告訴我們：「在狀態 $S$，我**預期**能拿到多少分？」

**2. 引入「優勢 (Advantage)」(高品質的學習訊號)**
* 我們現在可以計算一個高品質的「相對分數」，稱為「**優勢 (Advantage, $A_t$)**」。
* $\text{Advantage} = \text{實際拿到的分數} - \text{Critic 預期的分數}$
* $A_t = R_t - V(s_t)$
* **訊號解讀**：
    * **$A_t > 0$ (例如 `+20`)**：太棒了！你的表現**比預期好 20 分**。這是一個**高品質的「獎勵」訊號**。
    * **$A_t < 0$ (例如 `-10`)**：糟糕！你的表現**比預期差 10 分**。這是一個**高品質的「懲罰」訊號**。

### 5.2 Actor-Critic (演員-評論家) 架構

為了同時得到「策略」和「基線 (V值)」，我們需要**兩個**神經網路：

1.  **演員 (The Actor) - (策略網路 $\pi_{\theta}$)**
    * **工作**：做決策 (輸出動作機率)。(例如：`policy_model`)
    * **學習**：使用「**Advantage ($A_t$)**」訊號來學習。

2.  **評論家 (The Critic) - (價值網路 $V_{\phi}$)**
    * **工作**：**只**負責「打分數」，提供「基線 $V(s_t)$」。
    * **學習**：使用「**TD 誤差**」來學習 (讓 $V(s_t)$ 盡可能接近 $R_t$)。

**PPO/GRPO** 就是 Actor-Critic 家族的最新成員。

### 5.3 PPO (Proximal Policy Optimization) 核心理論

**Actor-Critic 的新問題**：訓練不穩定。Actor 根據一個「可能還很菜」的 Critic 給出的 Advantage，一次更新「**用力過猛**」，導致好不容易學會的策略**崩潰**。

**PPO 的解決方案**：增加「**安全鎖 (Safety Locks)**」，限制 Actor 每次更新的步伐。

**安全鎖 #1：PPO-Clip (超參數 `EPSILON = 0.2`)**
* **理論**：這是一個「**硬限制**」的裁剪 (Clipping) 方式。
* **比例 (Ratio)**： $\text{Ratio} = \frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$ (新策略機率 / 舊策略機率)
* **裁剪 (Clip)**：PPO **強迫** `Ratio` 必須被限制在 `[1 - EPSILON, 1 + EPSILON]` (例如 `[0.8, 1.2]`) 的範圍內。
* **Loss 函數**：PPO 會在「未裁剪的獎勵」和「裁剪後的獎勵」中，取**較小 (minimum)** 的那個。
    * `Loss = min(Adv * Ratio, Adv * clip(Ratio, 1-ε, 1+ε))`
* **結論**：`EPSILON` 確保 Actor 即使在看到一個巨大的 Advantage 時，也**不能**一次將策略更新超過 20%。

**安全鎖 #2：KL 散度懲罰 (超參數 `BETA` & `reference_model`)**
* **理論**：這是一個「**軟限制**」的懲罰 (Penalty) 方式。
* **`reference_model`**：就是「**舊策略 ($\pi_{\text{old}}$)**」的快照。
* **`KL 散度`**： $\text{KL}(\pi_{\text{new}} || \pi_{\text{old}})$，用來衡量「新舊策略的差異程度」。
* **Loss 函數**：$\text{Loss} = (\text{Advantage}) - (\text{BETA} \times \text{KL 散度})$
* **結論**：`BETA` 就像一條「**橡皮筋**」。Actor 可以自由更新，但如果它跑得離 `reference_model` 太遠 (`KL` 變大)，`BETA` 就會施加一個**懲罰**把它拉回來。

### 5.4 PPO 實作 (Flax NNX - Pendulum 專案)

我們最終實作了一個完整的 PPO Agent。

**1. `Actor(nnx.Module)` (演員)**
* 為了處理「連續動作」，網路輸出**兩個**頭：`mu` (平均值) 和 `sigma` (標準差)。
* `__call__` 函式回傳一個 `tfd.Normal` (常態分佈) 物件。

**2. `Critic(nnx.Module)` (評論家)**
* 一個標準的 MLP，`__call__` 函式回傳**一個**數字 (V 值 $V(s)$)。

**3. `PPOAgent(nnx.Module)`**
* **`__init__`**：初始化**兩個**網路 (`actor`, `critic`) 和**兩個** `nnx.Optimizer`。
* **`select_action` (JAX vs. NumPy 橋樑)**：
    1.  `state` (numpy) $\rightarrow$ `state_jnp` (jax, 增加 batch 維度)。
    2.  同時呼叫 `actor(state_jnp)` 和 `critic(state_jnp)`。
    3.  `dist = actor(...)` 取得機率分佈。
    4.  `action = dist.sample(...)` **採樣**動作 (探索)。
    5.  `log_prob = dist.log_prob(action)` 取得**「舊 Log 機率 ($\log \pi_{\text{old}}$)」** (PPO 學習**必須**)。
    6.  `value = critic(...)` 取得**「基線 ($V(s)$)」** (GAE 計算**必須**)。
    7.  將 `action`, `log_prob`, `value` 轉回 `numpy`。

**4. `RolloutBuffer` (On-Policy 儲存區)**
* **On-Policy (同策略)**：PPO **不能**使用 DQN 的 ReplayBuffer。
* PPO 的資料是「**易腐敗**的」，學完一次後必須**全部丟棄 (`clear()`)**。
* **`add(...)`**：儲存 `N` 步的 `(s, a, r, done, log_prob, value)`。
* **`calculate_advantages_and_returns(...)` (GAE 計算)**：
    * 這是最關鍵的數學。
    * 在 `N` 步收集完畢後，**從後往前 (reversed)** 遞迴計算：
    * **TD 誤差 (Delta)**：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
    * **優勢 (Advantage)**：$A_t = \delta_t + \gamma \lambda A_{t+1}$ (GAE 公式)
    * **回報 (Returns)**：$R_t = A_t + V(s_t)$ (Critic 的「正確答案」)
    * **優化**：最後將 `advantages` **標準化** `(adv - mean) / std`，使 Actor 學習更穩定。

### 5.4.1 GAE 詳細解釋 (最易混淆的部分！)

**為什麼需要 GAE？**

回顧優勢估計的兩個極端：

| 方法 | 公式 | 優點 | 缺點 |
|------|------|------|------|
| **1-step TD** | $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 低變異數 | 高偏差 (依賴 V 估計) |
| **Monte Carlo** | $A_t = G_t - V(s_t)$ | 無偏 | 高變異數 |

**GAE 的解決方案**：用 λ 參數在兩者間**插值**！

**完整 GAE 推導**：

定義 n-step advantage：
$$
A_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)
$$

GAE 是所有 n-step advantage 的**指數加權平均**：
$$
A_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} A_t^{(n)}
$$

**遞迴形式** (實作中使用)：
$$
A_t = \delta_t + (\gamma \lambda) A_{t+1}
$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**λ 參數的作用**：

* **λ = 0**：$A_t = \delta_t$ (純 1-step TD)
  * 低變異數，但完全依賴 V(s) 的準確性
* **λ = 1**：$A_t = G_t - V(s_t)$ (純 Monte Carlo)
  * 無偏，但高變異數
* **λ = 0.95** (常用值)：平衡偏差與變異數

**實作範例**：
```python
def calculate_advantages_and_returns(self, next_value):
    advantages = []
    returns = []
    gae = 0  # 初始化

    # 關鍵：從後往前遍歷！
    for t in reversed(range(len(self.rewards))):
        if t == len(self.rewards) - 1:
            next_value_t = next_value  # 最後一步用傳入的 next_value
        else:
            next_value_t = self.values[t + 1]

        # TD error
        delta = self.rewards[t] + GAMMA * next_value_t - self.values[t]

        # GAE 遞迴公式
        gae = delta + GAMMA * LAMBDA * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + self.values[t])  # R_t = A_t + V(s_t)

    # 標準化 advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages, returns
```

**為什麼要標準化 advantages？**
* 使不同 batch 的優勢值有相同的尺度
* 穩定 Actor 的梯度，避免學習率過大或過小
* 經驗上顯著提升訓練穩定性

**易錯點**：
1. ❌ 從前往後遍歷：會得到錯誤的 GAE
2. ❌ 忘記標準化：訓練會不穩定
3. ❌ 用 `append` 而非 `insert(0)`：順序錯誤

**5. `PPOAgent.train_step` (PPO 引擎室)**
* **`train_step`** 會被**反覆**呼叫 (K 個 Epochs)。
* **訓練 Critic**：
    * `values_pred = self.critic(batch_states)`
    * `loss = jnp.mean((values_pred - batch_returns) ** 2)` (標準 MSE Loss)
    * `grads` $\rightarrow$ `critic_optimizer.update(grads)`
* **訓練 Actor (PPO-Clip Loss)**：
    1.  **取得新機率**：`log_probs_new = self.actor(batch_states).log_prob(batch_actions)`
    2.  **計算 Ratio**：`ratio = jnp.exp(log_probs_new - batch_log_probs_old)`
    3.  **計算 Loss (Unclipped)**：`loss_unclipped = batch_advantages * ratio`
    4.  **計算 Loss (Clipped)**：`ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)`
    5.  **...** `loss_clipped = batch_advantages * ratio_clipped`
    6.  **取最小值**：`loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))` (加 `-` 號是為了「最小化」)
    7.  **更新**：`grads` $\rightarrow$ `actor_optimizer.update(grads)`

**6. `main()` (PPO 大迴圈)**
PPO 的生命週期就是一個「收集-學習-丟棄」的迴圈：
```python
while True:
    # --- 1. 收集 (Rollout) ---
    # 呼叫 N 次 agent.select_action()
    # 呼叫 N 次 buffer.add()

    # --- 2. 計算目標 (GAE) ---
    # 呼叫 buffer.calculate_advantages_and_returns()

    # --- 3. 學習 (Learn) ---
    # 呼叫 K 次 agent.train_step() (在 mini-batches 上)

    # --- 4. 丟棄 (Discard) ---
    # 呼叫 buffer.clear()
```

### 5.5 PPO vs DQN 完整對比

| 特性 | DQN | PPO |
|------|-----|-----|
| **學習目標** | Q 函數 $Q(s,a)$ | 策略 $\pi(a\|s)$ + 價值 $V(s)$ |
| **網路數量** | 2 (Online + Target) | 2 (Actor + Critic) |
| **資料收集** | Off-Policy | On-Policy |
| **資料儲存** | ReplayBuffer (重用) | RolloutBuffer (丟棄) |
| **連續動作** | ❌ 無法處理 | ✅ 完美支援 |
| **資料效率** | ✅ 高 (可重用舊資料) | ❌ 低 (必須用新資料) |
| **訓練穩定性** | 中等 (需 Target Network) | ✅ 高 (PPO-Clip) |
| **探索策略** | ε-greedy | 隨機策略內建探索 |
| **優勢估計** | N/A | GAE (λ) |
| **更新次數** | 每步更新 | 每 N 步更新 K epochs |
| **應用場景** | 遊戲 (Atari) | 機器人、LLM (RLHF) |
| **記憶體需求** | 中等 (Buffer) | 低 (只存 N 步) |

**為什麼 LLM 用 PPO 而非 DQN？**
1. LLM 的動作空間 = 詞彙表 (50K+ tokens)
2. 即使是離散的，DQN 需要對每個 token 計算 Q 值 (太慢)
3. PPO 直接輸出機率分佈，更自然

### 5.6 PPO 超參數調整指南

**關鍵超參數表**：

| 超參數 | 典型值 | 作用 | 調整建議 |
|--------|--------|------|----------|
| **CLIP_EPSILON** | 0.1 - 0.3 | 限制策略更新幅度 | 0.2 是最常用值 |
| **GAE_LAMBDA** | 0.9 - 0.99 | 偏差-變異數平衡 | 稀疏獎勵用 0.99 |
| **GAMMA** | 0.99 - 0.999 | 折扣因子 | 長期任務用 0.999 |
| **Learning Rate (Actor)** | 3e-4 | Actor 學習率 | 可以比 Critic 小 |
| **Learning Rate (Critic)** | 1e-3 | Critic 學習率 | - |
| **Rollout Steps** | 1024 - 4096 | 每次收集多少步 | 越大越穩定但慢 |
| **Epochs** | 3 - 10 | 重複訓練次數 | 太大會過擬合舊資料 |
| **Mini-batch Size** | 64 - 256 | 訓練批次大小 | - |

**調試流程**：
1. 先用預設值訓練，觀察是否學習
2. 如果策略崩潰 (獎勵突然下降)：
   - 降低學習率
   - 降低 CLIP_EPSILON
   - 減少 Epochs
3. 如果學習太慢：
   - 增加 Rollout Steps
   - 增加 Learning Rate
4. 如果獎勵不穩定：
   - 確認 advantages 有標準化
   - 檢查 GAE 實作是否正確

**常見錯誤與解決**：

| 問題 | 症狀 | 原因 | 解決 |
|------|------|------|------|
| **策略崩潰** | 獎勵突然歸零 | 更新步長太大 | 降低 LR 或 CLIP_EPSILON |
| **不學習** | 獎勵不上升 | Critic 太差 | 增加 Critic LR |
| **過擬合** | 訓練好測試差 | Epochs 太多 | 減少到 3-5 |
| **高方差** | 獎勵劇烈震盪 | λ 太大 | 降低 GAE_LAMBDA |

### 5.7 為什麼需要「舊」log_prob？(重要性採樣)

這是學生最常問的問題！

**回顧 PPO-Clip loss**：
```python
ratio = exp(log_prob_new - log_prob_old)  # π_new(a|s) / π_old(a|s)
loss = min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)
```

**為什麼需要 `log_prob_old`？**

因為我們在做**重要性採樣 (Importance Sampling)**！

**問題來源**：
* 資料是用 $\pi_{\text{old}}$ 收集的
* 但我們要用這些資料來訓練 $\pi_{\text{new}}$
* 這兩個分佈不同！

**重要性採樣修正**：
$$
\mathbb{E}_{a \sim \pi_{\text{new}}}[f(a)] = \mathbb{E}_{a \sim \pi_{\text{old}}}\left[ \frac{\pi_{\text{new}}(a)}{\pi_{\text{old}}(a)} f(a) \right]
$$

這個 $\frac{\pi_{\text{new}}(a)}{\pi_{\text{old}}(a)}$ 就是 `ratio`！

**PPO 的創新**：
* 如果 ratio 太大 (新舊策略差太多)，就 clip 住
* 防止重要性採樣的高變異數問題

**實作中的關鍵**：
```python
# 在 select_action 時保存 log_prob_old
log_prob_old = actor(state).log_prob(action)
buffer.add(state, action, reward, done, log_prob_old, value)

# 訓練時計算 log_prob_new
log_prob_new = actor(state).log_prob(action)  # 用更新後的 actor
ratio = exp(log_prob_new - log_prob_old)
```

📁 **完整實作程式碼**：[`5.1.pendulum/pendulum.py`](5.1.pendulum/pendulum.py)
📖 **詳細說明文件**：[`5.1.pendulum/README.md`](5.1.pendulum/README.md)

---

[⬅️ 上一章：模組四 - Policy Gradients](../4.Policy_Gradients/README.md) | [返回目錄](../README.md) | [下一章：模組六 - JAX/Maxtext ➡️](../6.Jax&Maxtext/README.md)