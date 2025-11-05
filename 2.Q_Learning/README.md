## <a name="module-2-tabular-methods-q-learning"></a>模組二：表格型解法 (Q-Learning)

[⬅️ 上一章：模組一 - RL 的核心概念](../1.Basics/README.md) | [返回目錄](../README.md) | [下一章：模組三 - DQN ➡️](../3.DQN/README.md)

---

本模組介紹了第一個具體的 RL 演算法，適用於「狀態空間」很小 (例如 4x4 網格) 的問題。

### 2.1 Q-Function (Q 函數)

Q-Learning 引入了一個比 $V(s)$ 更強大的「**Q 函數**」，也叫「**Q 值**」。

**從 V 到 Q 的演進**：
* $V(s)$：「在狀態 $s$」有多好？
* $Q(s, a)$：「在狀態 $s$，**並且**執行動作 $a$」有多好？

**為什麼 Q 函數更強大？**

如果只有 $V(s)$，Agent 需要知道「環境的動態模型」才能做決策：
```
選擇動作 a = argmax_a [ R(s,a) + γ·V(s') ]
                         ↑需要知道執行 a 後會到達哪個 s'
```

但如果有 $Q(s,a)$，決策變得超級簡單：
```
選擇動作 a = argmax_a Q(s, a)  ← 直接比大小！
```

**Q 函數的完整定義**：
$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[ G_t \mid S_t = s, A_t = a \right]
$$

意思是：「在狀態 $s$ 執行動作 $a$，然後遵循策略 $\pi$，預期能獲得的總回報。」

**Q 與 V 的關係**：
$$
V^{\pi}(s) = \sum_{a} \pi(a|s) \cdot Q^{\pi}(s, a)
$$
$$
Q^{\pi}(s, a) = \mathbb{E}\left[ R + \gamma V^{\pi}(S') \right]
$$

**Q-Table 實作範例**：

在表格型解法中，我們用一個「**Q-Table (Q 表格)**」 (例如 NumPy 陣列) 來儲存**每一個** $(s, a)$ 組合的 Q 值。

```python
# 4x4 網格世界 (16 個狀態)，4 個動作 (上下左右)
q_table = np.zeros((16, 4))

# q_table[state][action] = Q 值
# 例如：q_table[5][2] 表示「在狀態 5，執行動作 2」的 Q 值

# 動作選擇 (貪婪策略)
action = np.argmax(q_table[state])  # 選擇 Q 值最高的動作
```

### 2.2 時間差分學習 (TD Learning)

Q-Learning 是一種「**時間差分 (Temporal-Difference, TD) 學習**」方法。

**核心思想**：我們**不需要**等到遊戲結束，才回頭更新價值。我們在「**每走一步**」時，就「**用未來的現實，修正過去的預估**」。

**三種學習方式對比**：

| 方法 | 更新時機 | 更新公式 | 優點 | 缺點 |
|------|----------|----------|------|------|
| **Monte Carlo (MC)** | 回合結束後 | $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$ | 無偏估計 | 高變異數、需等待回合結束 |
| **TD(0)** | 每一步 | $V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]$ | 低變異數、立即學習 | 有偏估計 (bootstrap) |
| **TD(λ)** | 每一步 | 介於 MC 和 TD(0) 之間 | 平衡偏差與變異數 | 實作複雜 |

**TD Learning 的直覺例子**：
* 你預估上班要 30 分鐘。5 分鐘後，你看到高速公路堵死了。你**立刻** (TD Learning) 更新了預估 (變成 2 小時)，而**不是** 2 小時後才「學到」這件事。

**Bootstrap (自舉) 的意義**：
TD 方法用「一個估計值」($V(s')$) 去更新「另一個估計值」($V(s)$)，這叫做「自舉」。
* 優點：可以立即學習，不需要等到回合結束
* 缺點：如果 $V(s')$ 的估計不準，錯誤會傳播

### 2.3 Q-Learning 演算法 (Q-Table 更新)

Q-Learning 的目標就是學習這個 Q-Table。在 Agent 執行了 `(S, A, R, S')` 這一組經驗後，它會用「TD Learning」來更新 Q-Table：

**1. 計算「TD 目標」(Target) - (即「更準確的現實」)**
$$
\text{TD Target} = R + \gamma \cdot \max_{a'} Q(S', a')
$$
* **$R$**：你**立刻**拿到的獎勵。
* **$\gamma$ (gamma)**：折扣因子 (例如 0.99)，代表對未來獎勵的重視程度。
* **$\max_{a'} Q(S', a')$**：Agent 查看 Q-Table，找出「**在新狀態 $S'$**」下，「**最好的下一步**」的 Q 值。

**2. 計算「TD 誤差」(Error)**
$$
\text{TD Error} = \text{TD Target} - Q(S, A)
$$
* $\text{TD Error}$ 就是「新現實」和「舊預估」之間的差距。

**3. 更新 Q-Table**
$$
Q(S, A) \leftarrow Q(S, A) + \alpha \cdot (\text{TD Error})
$$
* **$\alpha$ (alpha)**：學習率 (例如 0.1)，代表你這次要「相信」這個誤差多少。
* **邏輯**：將「舊的 Q 值」往「更準確的 TD Target」方向**移動一點點**。

**完整更新公式 (合併)**：
$$
Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right]
$$

**Python 實作範例**：
```python
# 執行動作，獲得經驗
next_state, reward, done = env.step(action)

# Q-Learning 更新
td_target = reward + GAMMA * np.max(q_table[next_state])
td_error = td_target - q_table[state, action]
q_table[state, action] += ALPHA * td_error
```

### 2.4 Off-Policy vs On-Policy (關鍵概念！)

這是 RL 中最重要的分類之一，直接影響後續 DQN vs PPO 的理解。

**Q-Learning 是 Off-Policy 方法**：
* **行為策略 (Behavior Policy)**：Agent 實際用來採取動作的策略 (例如 ε-greedy)
* **目標策略 (Target Policy)**：我們想要學習的策略 (貪婪策略，$\arg\max_a Q(s,a)$)

**Off-Policy 的關鍵特性**：
$$
\text{TD Target} = R + \gamma \max_{a'} Q(S', a')  \quad \leftarrow \text{這裡用 } \max \text{！}
$$

注意這個 $\max$ 操作：
* Agent 實際上可能用 ε-greedy 選了一個「隨機」動作
* 但更新時，我們用「**最佳**動作」的 Q 值來計算 TD Target
* 行為策略 ≠ 目標策略 → Off-Policy

**對比：SARSA (On-Policy)**：
$$
\text{TD Target} = R + \gamma Q(S', A')  \quad \leftarrow \text{這裡用實際採取的動作 } A' \text{！}
$$

**Off-Policy vs On-Policy 對比表**：

| 特性 | Off-Policy (Q-Learning) | On-Policy (SARSA, PPO) |
|------|-------------------------|------------------------|
| **行為策略 = 目標策略？** | ❌ 不同 | ✅ 相同 |
| **更新使用** | 最優動作 ($\max$) | 實際動作 ($A'$) |
| **資料效率** | ✅ 高 (可重用舊資料) | ❌ 低 (只能用新資料) |
| **收斂性** | 保證收斂到最優 | 收斂到當前策略的最優 |
| **探索影響** | 不影響學習目標 | 直接影響學習 |
| **代表演算法** | Q-Learning, DQN | SARSA, PPO, GRPO |

**為什麼 Off-Policy 可以重用舊資料？**
* 因為 Q-Learning 學的是「**最優** Q 函數」，與「**怎麼收集資料**」無關
* 即使資料是用很爛的策略收集的，只要包含 $(s, a, r, s')$，就能學習
* 這是 DQN 的 Experience Replay 能成功的理論基礎！

### 2.5 收斂性與超參數調整

**Q-Learning 收斂保證 (Watkins & Dayan, 1992)**：

在以下條件下，Q-Learning 保證收斂到最優 Q 函數 $Q^*$：
1. **所有 $(s, a)$ pair 被無限次訪問**：充分探索
2. **學習率 $\alpha$ 滿足 Robbins-Monro 條件**：
   $$
   \sum_{t=1}^{\infty} \alpha_t = \infty, \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty
   $$
   * 例如：$\alpha_t = \frac{1}{t}$ 或 $\alpha_t = \frac{1}{1 + t}$

**實務中的固定學習率**：
* 理論要求 $\alpha$ 隨時間衰減，但實務中常用固定值 (例如 0.1)
* 好處：在非平穩環境中能持續適應
* 代價：永遠不會完全收斂 (會在最優解附近震盪)

**關鍵超參數表**：

| 超參數 | 典型值 | 作用 | 調整指南 |
|--------|--------|------|----------|
| **$\alpha$ (學習率)** | 0.01 - 0.5 | 控制更新步長 | 太大：震盪；太小：收斂慢 |
| **$\gamma$ (折扣因子)** | 0.9 - 0.99 | 控制時間視野 | 稀疏獎勵用 0.99；密集用 0.9 |
| **$\epsilon$ (探索率)** | 0.1 - 0.3 | 探索-利用平衡 | 初期 1.0 → 後期 0.01 (衰減) |

**常見問題與調試**：
* **Q 值全部是 0**：探索不足，增加 $\epsilon$ 或訓練時間
* **Q 值爆炸**：學習率 $\alpha$ 太大，或折扣因子 $\gamma \geq 1$
* **不收斂**：狀態空間太大 (該用 DQN)，或探索策略不當

**實作 (NumPy)**：
我們在 4x4 網格世界中，使用 NumPy 陣列 `q_table = np.zeros((16, 4))` 來實作了這個演算法。

📁 **完整實作程式碼**：[`2.1.Q_Learning/Q_Learning.py`](2.1.Q_Learning/Q_Learning.py)
📖 **詳細說明文件**：[`2.1.Q_Learning/README.md`](2.1.Q_Learning/README.md)

---

[⬅️ 上一章：模組一 - RL 的核心概念](../1.Basics/README.md) | [返回目錄](../README.md) | [下一章：模組三 - DQN ➡️](../3.DQN/README.md)