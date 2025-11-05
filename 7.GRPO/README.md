## 模組七：GRPO (Group Relative Policy Optimization) 專案

[⬅️ 上一章：模組六 - JAX/Maxtext](../6.Jax&Maxtext/README.md) | [返回目錄](../README.md)

---

本模組是一個**畢業專案**，旨在將 PPO 的「安全鎖」理論，與一種更高效的、**無評論家 (Critic-less)** 的基線計算方法相結合。

這**完美地**對應了現代 LLM 訓練 (如 `GRPO on Maxtext`) 的核心思想：**為了節省記憶體而移除 `Critic` 網路**。

### 7.1 GRPO 的「核心取捨 (Trade-off)」

**1. 標準 PPO (我們在模組五的實作)**
* **Actor (演員) $\pi_{\theta}$**：一個神經網路。
* **Critic (評論家) $V_{\phi}$**：**另一個**神經網路。
* **優勢 (Advantage)**：$A_t = (\text{實際總分 } R_t) - (\text{Critic 預期分數 } V_{\phi}(s_t))$
* **問題**：訓練 `Critic` ( $V_{\phi}$ ) 需要**額外大量的記憶體** (模型權重 + 梯度 + Adam 狀態)，這在訓練 LLM 這種 80 億 (8B) 參數的模型時是**難以承受**的。

**2. GRPO (Group Relative) 的「革命」**
* **GRPO 的核心思想**：我們**完全移除 `Critic` 網路**。
* **GRPO 如何計算「基線 (Baseline)」？**
    * 它**不**學習 (learn) 基線，而是**動態計算 (calculate)** 基線。
    * 它依賴於一個「**群組 (Group)**」的**平均表現**。
* **GRPO 的「相對優勢 (Relative Advantage)」**：
    * $\text{Advantage}_i = R_i - \frac{1}{G}\sum_{j=1}^{G} R_j$

**關鍵：群組 (Group) 的定義**
* **不是**所有訓練樣本的平均
* **是**「同一個 prompt 的 G 個不同生成」的平均
* 例如：給定問題「1+1=?」，生成 G=4 個答案：
  ```
  答案1: "2" → 獎勵 R₁ = +1.0 ✅
  答案2: "3" → 獎勵 R₂ = 0.0  ❌
  答案3: "2" → 獎勵 R₃ = +1.0 ✅
  答案4: "4" → 獎勵 R₄ = 0.0  ❌

  群組平均 = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5

  Advantage₁ = 1.0 - 0.5 = +0.5 (獎勵)
  Advantage₂ = 0.0 - 0.5 = -0.5 (懲罰)
  Advantage₃ = 1.0 - 0.5 = +0.5 (獎勵)
  Advantage₄ = 0.0 - 0.5 = -0.5 (懲罰)
  ```

**這個設計的巧妙之處**：
* 「2」比群組平均好 → 增加機率
* 「3」和「4」比群組平均差 → 降低機率
* 不需要 Critic 網路來估計「"2" 這個答案有多好」
* 只需要相對比較！

**3. GRPO 的「成本轉移」
* **優點 (Pro)**：
    * **節省記憶體**：省下了一個 8B 參數 `Critic` 網路所需的所有記憶體。
* **缺點 (Con)**：
    * **收集成本更高**：為了計算「群組平均」，你必須為**同一個**狀態 $S$ **生成 `G` 次** ( `NUM_GENERATIONS` ) 回答，這使得「資料收集 (Rollout)」階段的計算成本變成了 `G` 倍。
* **結論**：GRPO 是一種**犧牲「收集資料」的效率，來換取「訓練」時記憶體效率**的演算法。

---

### 7.2 專案實作：GRPO on MNIST (真實資料)

我們將這個「LLM 演算法」應用到了一個「影像分類」問題上，以驗證其核心邏R輯。

**1. 將「分類」問題 RL 化**
* **狀態 (State, $s$)**：一張 `(28, 28)` 的 MNIST 圖片 (扁平化為 `(784,)`)。
* **動作 (Action, $a$)**：`Actor` 網路**猜測**的數字 (0-9)。
* **這是一個「單步 (One-Step)」回合**：猜完遊戲就結束。
* **獎勵 (Reward, $r$)**：答對 `+1.0`，答錯 `0.0`。

**2. GRPO 的核心實作**
* **Actor 網路**：`Input(784) → 128 → 128 → Output(10)`，輸出一個 `Categorical` 分佈。
* **無 Critic 網路**：這是 GRPO 的關鍵特色。
* **群組基線計算**：
    ```python
    # 為一批 (G=1024) 張圖片生成預測
    actions = actor.sample(batch_images)
    rewards = (actions == labels).astype(float)

    # 計算群組平均作為基線 (取代 Critic)
    baseline = mean(rewards)

    # 計算相對優勢
    advantages = rewards - baseline
    ```
* **PPO-Clip 訓練**：使用與模組五相同的 PPO-Clip Loss 來訓練 Actor。

**3. 記憶體節省效果**
* **PPO**: 需要 Actor + Critic 兩個網路
* **GRPO**: 只需要 Actor 一個網路
* **節省**: 約 50% 的參數量 (在 LLM 場景下節省更加顯著)

**4. 預期結果**
* 經過 10 個 Epochs 的訓練，準確率應達到 95% 以上
* 證明了 GRPO 可以成功應用於監督式學習問題

📁 **完整實作程式碼**：[`7.1.GRPO_MNIST/grpo_mnist.py`](7.1.GRPO_MNIST/grpo_mnist.py)
📖 **詳細說明文件**：[`7.1.GRPO_MNIST/README.md`](7.1.GRPO_MNIST/README.md)

📁 **GRPO on MaxText 實作**：[`7.2.GRPO_Maxtext/grpo_llama3_1_8b_demo_pw.py`](7.2.GRPO_Maxtext/grpo_llama3_1_8b_demo_pw.py)
📖 **MaxText 詳細說明文件**：[`7.2.GRPO_Maxtext/README.md`](7.2.GRPO_Maxtext/README.md)

### 7.3 GRPO vs PPO 完整對比

| 特性 | PPO | GRPO |
|------|-----|------|
| **網路結構** | Actor + Critic (2個網路) | Actor only (1個網路) |
| **Baseline 來源** | Critic $V(s)$ (學習) | 群組平均 $\bar{R}$ (計算) |
| **Advantage 計算** | GAE: $A_t = \delta_t + \gamma\lambda A_{t+1}$ | Group Mean: $A_i = R_i - \bar{R}$ |
| **記憶體需求** | 高 (Actor + Critic) | ✅ 低 (只 Actor) |
| **資料收集成本** | 正常 (每 state 採樣 1 次) | ❌ 高 (每 prompt 生成 G 次) |
| **訓練穩定性** | 高 (Critic 提供穩定 baseline) | 中等 (依賴 G 的大小) |
| **適用場景** | 通用 RL | ✅ LLM RLHF (記憶體受限) |
| **Group Size (G)** | N/A | 需調整 (典型 64-128) |

**為什麼 LLM 訓練偏好 GRPO？**

假設訓練 Llama 3.1 8B：
* **PPO 記憶體需求**：
  - Actor (8B 參數): 16 GB (權重) + 16 GB (梯度) + 32 GB (Adam) = 64 GB
  - Critic (8B 參數): 64 GB
  - **總計：128 GB**

* **GRPO 記憶體需求**：
  - Actor (8B 參數): 64 GB
  - Critic: 0 GB
  - **總計：64 GB** (省一半！)

**權衡**：
* 用 vLLM 等高效推理引擎加速「生成 G 次」
* 生成成本 << 訓練 Critic 的記憶體成本

### 7.4 Group Size (G) 的選擇

**G 太小的問題**：
```python
# G = 2 (太小)
prompt = "1+1=?"
R₁ = +1.0  # 正確答案 "2"
R₂ = 0.0   # 錯誤答案 "3"

baseline = (1.0 + 0.0) / 2 = 0.5
Adv₁ = 1.0 - 0.5 = +0.5
Adv₂ = 0.0 - 0.5 = -0.5

# 問題：baseline 不穩定 (只有2個樣本)
```

**G 太大的問題**：
```python
# G = 512 (太大)
# 每個 prompt 需要生成 512 次
# 如果有 1000 個 prompts → 需要 512,000 次生成！
# 計算成本爆炸
```

**實務建議**：

| 場景 | 推薦 G | 原因 |
|------|--------|------|
| **MNIST (簡單)** | 32 - 128 | 獎勵簡單 (0/1) |
| **LLM 數學題** | 64 - 128 | 獎勵明確 |
| **LLM 開放生成** | 128 - 256 | 需要更多樣本穩定 baseline |
| **記憶體受限** | 越小越好 | 減少推理成本 |

**調整策略**：
1. 從 G=64 開始
2. 監控 advantage 的標準差：
   - 太大 (>1.0)：增加 G
   - 穩定 (<0.5)：可以減少 G
3. 平衡記憶體與訓練穩定性

### 7.5 GRPO 在 LLM RLHF 的完整流程

**典型 LLM RLHF 流程 (GRPO)**：

```python
# 1. 準備 prompts 和 reward model
prompts = ["請解釋量子力學", "1+1=?", ...]
reward_model = load_reward_model()  # 人類偏好模型

# 2. 對每個 prompt 生成 G 個回答
for prompt in prompts:
    responses = []
    rewards = []

    for _ in range(G):  # G = 64
        response = actor_model.generate(prompt)
        reward = reward_model.score(prompt, response)
        responses.append(response)
        rewards.append(reward)

    # 3. 計算群組 baseline
    baseline = np.mean(rewards)

    # 4. 計算 advantages
    advantages = [r - baseline for r in rewards]

    # 5. PPO-Clip 訓練
    for response, advantage in zip(responses, advantages):
        log_prob_old = actor_model.log_prob(prompt, response)
        # ... (標準 PPO 訓練)
```

**與標準 PPO RLHF 的差異**：
```python
# 標準 PPO RLHF
response = actor.generate(prompt)  # 只生成 1 次
reward = reward_model.score(prompt, response)
value = critic.forward(prompt)  # 用 Critic 估計 baseline
advantage = reward - value

# GRPO
responses = [actor.generate(prompt) for _ in range(G)]  # 生成 G 次
rewards = [reward_model.score(prompt, r) for r in responses]
baseline = np.mean(rewards)  # 用群組平均當 baseline
advantages = [r - baseline for r in rewards]
```

### 7.6 易混淆點總結

**❌ 常見誤解 1**：「GRPO 就是去掉 Critic 的 PPO」
* **部分正確**，但重點是「如何計算 baseline」
* 不是不用 baseline，而是用「群組平均」取代「Critic 網路」

**❌ 常見誤解 2**：「G 是 batch size」
* **錯誤**！G 是「同一個 prompt 的生成次數」
* Batch size 是「多少個不同 prompts」
* 例如：Batch=32 prompts，每個生成 G=64 次 → 總共 2048 個樣本

**❌ 常見誤解 3**：「GRPO 比 PPO 好」
* **不一定**！GRPO 是記憶體與計算的權衡
* 如果記憶體足夠，PPO 可能更穩定 (Critic 提供更好的 baseline)
* GRPO 的優勢在於「無法負擔 Critic 的場景」

**✅ 關鍵洞察**：
* GRPO 是工程需求驅動的演算法設計
* 核心理念：用「計算」換「記憶體」
* 在 LLM 時代特別有用 (模型太大，記憶體吃緊)

---

[⬅️ 上一章：模組六 - JAX/Maxtext](../6.Jax&Maxtext/README.md) | [返回目錄](../README.md)