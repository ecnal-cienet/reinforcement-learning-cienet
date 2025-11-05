## <a name="module-4-policy-gradients-the-why"></a>模組四：策略梯度 (Policy Gradients) - (PPO/GRPO 的基石)

[⬅️ 上一章：模組三 - DQN](../3.DQN/README.md) | [返回目錄](../README.md) | [下一章：模組五 - PPO ➡️](../5.PPO/README.md)

---

本模組介紹了 RL 的**第二大家族**：「策略為基礎 (Policy-Based)」的方法。這是你工作中 `PPO/GRPO` 演算法的**直系祖先**。

### 4.1 DQN (價值為基礎) 的局限性

DQN (模組三) 非常強大，但有致命缺陷：
1.  **無法處理連續動作 (Continuous Actions)**：DQN 依賴 $\max_{a'}$ 操作來選擇動作。如果動作是「方向盤角度」(一個 -180 到 +180 的連續數字)，你無法對「無限多個」動作取 $\max$。
2.  **不夠直接**：我們真正想要的，是「策略 (Policy)」本身。DQN 只能「間接」地從 Q 值中「推導」出策略。

### 4.2 策略梯度 (Policy Gradient) 演算法

**核心思想**：我們**直接**學習「策略 (Policy)」本身。

我們建立一個「**策略網路 (Policy Network)**」，用 $\pi_{\theta}$ 表示 (其中 $\theta$ 是神經網路權重)。
* **輸入 (Input)**：狀態 $S$。
* **輸出 (Output)**：一個**機率分佈**，代表所有動作的機率。
    * **離散 (CartPole)**：`[P(左), P(右)]` (例如 `[0.7, 0.3]`)
    * **連續 (Llama LLM)**：`[P(詞A), P(詞B), ...]` (所有詞彙的機率)
    * **連續 (Pendulum)**：一個機率分佈的**參數** (例如：平均值 $\mu=0.5$，標準差 $\sigma=0.1$)。

**策略梯度定理 (Policy Gradient Theorem)**

這是策略梯度方法的數學基礎：

**目標函數**：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]
$$
其中 $\tau$ 是一條軌跡 (trajectory)：$(s_0, a_0, r_0, s_1, a_1, r_1, ...)$

**梯度公式** (Sutton et al., 1999)：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot G_t \right]
$$

**直覺理解**：
* $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$：「如何調整權重 $\theta$ 來**增加**動作 $a_t$ 的機率」
* $G_t$：「這個動作帶來的回報有多好」
* 乘積：「**好的動作**增加機率，**壞的動作**減少機率」

**為什麼可以處理連續動作？**

離散動作 (Categorical)：
```python
logits = policy_network(state)  # [2.1, 0.5, -1.3]
probs = softmax(logits)         # [0.7, 0.2, 0.1]
action = sample(probs)          # 採樣 → 0
log_prob = log(probs[action])   # log(0.7) = -0.357
```

連續動作 (Normal distribution)：
```python
mu, sigma = policy_network(state)  # μ=0.5, σ=0.1
dist = Normal(mu, sigma)
action = dist.sample()             # 採樣 → 0.52
log_prob = dist.log_prob(action)   # log p(0.52|μ=0.5,σ=0.1)
```

**關鍵**：兩者都能計算 `log_prob(action)`！

**學習邏輯：REINFORCE (蒙地卡羅法)**

**完整演算法**：
```
For each episode:
    1. 收集軌跡 τ = (s₀,a₀,r₀, s₁,a₁,r₁, ..., s_T)
    2. For t = 0 to T:
           計算 G_t = Σ(γᵏ r_{t+k})  # 從 t 到結束的折扣回報
    3. For t = 0 to T:
           loss += -log π_θ(a_t|s_t) · G_t
    4. θ ← θ - α · ∇_θ loss
```

**Python 實作框架**：
```python
# 收集一個 episode
states, actions, rewards = [], [], []
for t in range(episode_length):
    action = policy_network.sample(state)
    next_state, reward, done = env.step(action)
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    if done: break

# 計算 returns
returns = []
G = 0
for r in reversed(rewards):
    G = r + GAMMA * G
    returns.insert(0, G)

# 更新策略
for s, a, G in zip(states, actions, returns):
    log_prob = policy_network.log_prob(s, a)
    loss = -log_prob * G  # 負號因為要做梯度上升
    optimizer.step(loss)
```

### 4.3 REINFORCE 的致命缺陷

這個簡單的演算法有兩個你親自發現的致命缺陷：

1.  **高變異數 (High Variance) / 信用分配 (Credit Assignment) 問題**：
    * REINFORCE 就像一個「**只看團隊總分**」的教授。
    * 一個 A+ 的團隊專案 (好遊戲)，會讓團隊中**「擺爛」**的成員 (壞動作) **也得到獎勵**。
    * 一個 F- 的團隊專案 (壞遊戲)，會讓團隊中**「努力」**的成員 (好動作) **也受到懲罰**。
    * 這個學習訊號**雜訊非常高 (noisy)**。

2.  **缺乏「標準」 (No Baseline) 問題**：
    * 你提出的關鍵問題：「為什麼 `+200` 分算好？」
    * 如果平均成績是 `+50`，`+200` 就是 A+。
    * 如果平均成績是 `+400`，`+200` 就是 F-。
    * 「**絕對總分 ($R_{\text{total}}$)**」本身是一個**毫無意義**的學習訊號。

### 4.4 從 REINFORCE 到 Actor-Critic 的演進

**問題 1 的解決：引入 Baseline**

修改梯度公式：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}\left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot (G_t - b(s_t)) \right]
$$

其中 $b(s_t)$ 是 baseline (基線)。

**數學保證**：加入 baseline 不會改變期望梯度，但會**降低變異數**！

**最佳 baseline**：$b(s_t) = V^{\pi}(s_t)$ (狀態價值函數)

這樣 $(G_t - V(s_t))$ 就是**優勢 (Advantage)**！

**問題 2 的解決：用 TD 取代 MC**

REINFORCE 用 Monte Carlo：
$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \quad \text{(高變異數)}
$$

Actor-Critic 用 TD：
$$
A_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{(低變異數)}
$$

**這就是 Actor-Critic 的核心改進！**

### 4.5 策略梯度方法對比表

| 方法 | Return 估計 | Baseline | 變異數 | 偏差 | 適用場景 |
|------|-------------|----------|--------|------|----------|
| **REINFORCE** | Monte Carlo $G_t$ | ❌ 無 | 極高 | 無偏 | 簡單任務 |
| **REINFORCE w/ Baseline** | Monte Carlo $G_t$ | ✅ $V(s)$ | 高 | 無偏 | 中等任務 |
| **Actor-Critic** | TD $r + \gamma V(s')$ | ✅ $V(s)$ | 低 | 有偏 | 大多數任務 |
| **A2C/A3C** | n-step TD | ✅ $V(s)$ | 中 | 低偏 | 並行訓練 |
| **PPO** | GAE (λ-return) | ✅ $V(s)$ | 低 | 低偏 | 生產級 |

**演進路徑**：
```
REINFORCE (1992)
    ↓ + Baseline
REINFORCE with Baseline
    ↓ + TD Learning
Actor-Critic (Sutton, 1984)
    ↓ + GAE
A3C (2016) / A2C
    ↓ + PPO-Clip
PPO (2017) ← 你的工作!
    ↓ - Critic
GRPO (2024) ← LLM 訓練!
```

### 4.6 關鍵洞察：On-Policy 的必然性

**為什麼策略梯度方法都是 On-Policy？**

回顧梯度公式：
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[ ... \right]
$$

注意：期望是在 $\pi_{\theta}$ 下計算的！

**這意味著**：
* 資料必須用**當前策略 $\pi_{\theta}$** 收集
* 一旦 $\theta$ 更新，舊資料就不能用了
* 這就是為什麼 PPO/GRPO 不能用 ReplayBuffer！

**對比 DQN (Off-Policy)**：
* DQN 學習 $Q^*(s,a)$ (最優 Q 函數)
* $Q^*$ 與「怎麼收集資料」無關
* 所以可以重用舊資料

**重要性採樣 (Importance Sampling) 的嘗試**：
* 可以用 importance sampling 修正分佈差異
* 但會引入高變異數 (PPO 用 Clip 解決)
* 詳見模組五

---

[⬅️ 上一章：模組三 - DQN](../3.DQN/README.md) | [返回目錄](../README.md) | [下一章：模組五 - PPO ➡️](../5.PPO/README.md)