# Q-Learning 實作 (表格型強化學習)

## 概述

這是一個經典的 **Q-Learning** 演算法實作，應用於簡單的 **4x4 網格世界 (Grid World)** 環境。Q-Learning 是一種基於價值 (Value-Based) 的無模型 (Model-Free) 強化學習演算法，特別適合狀態空間較小的離散環境。

## 環境說明

### 網格世界 (Grid World)

```
狀態編號排列：
┌────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │  起點: 狀態 0 (左上角)
├────┼────┼────┼────┤
│ 4  │ 5  │ 6  │ 7  │
├────┼────┼────┼────┤
│ 8  │ 9  │ 10 │ 11 │
├────┼────┼────┼────┤
│ 12 │ 13 │ 14 │ 15 │  目標: 狀態 15 (右下角)
└────┴────┴────┴────┘
```

### 環境參數

- **狀態空間 (State Space)**：16 個離散狀態 (0 到 15)
- **動作空間 (Action Space)**：4 個動作
  - `0`: 上 (↑)
  - `1`: 下 (↓)
  - `2`: 左 (←)
  - `3`: 右 (→)
- **起點 (Start State)**：狀態 0 (左上角)
- **目標 (Goal State)**：狀態 15 (右下角)

### 獎勵函數 (Reward Function)

- **到達目標狀態 (狀態 15)**：獎勵 `+100`
- **每移動一步**：扣 `-1` (鼓勵找到最短路徑)

## 執行方式

### 前置條件

確保已啟動虛擬環境並安裝相依套件：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 執行程式

```bash
python 2.Q_Learning/2.1.Q_Learning/Q_Learning.py
```

或者在 `2.Q_Learning/2.1.Q_Learning` 目錄內執行：

```bash
cd 2.Q_Learning/2.1.Q_Learning
python Q_Learning.py
```

## 演算法核心

### Q-Learning 更新公式

Q-Learning 使用 **時間差分 (Temporal-Difference, TD)** 方法來更新 Q-Table：

```
Q(S, A) ← Q(S, A) + α × [R + γ × max Q(S', a') - Q(S, A)]
                              └─────────────────┘
                                  TD Target
```

**參數說明：**
- **S**: 當前狀態
- **A**: 執行的動作
- **R**: 獲得的即時獎勵
- **S'**: 新狀態
- **α (alpha)**: 學習率 = `0.1` (控制對新資訊的信任程度)
- **γ (gamma)**: 折扣因子 = `0.99` (控制對未來獎勵的重視程度)

### Epsilon-Greedy 策略

在訓練過程中，Agent 使用 **Epsilon-Greedy** 策略來平衡探索 (Exploration) 與利用 (Exploitation)：

- **探索** (機率 = ε)：隨機選擇動作
- **利用** (機率 = 1-ε)：選擇 Q 值最高的動作

**Epsilon 衰退：**
- **初始值 (EPSILON)**：`1.0` (100% 探索)
- **最終值 (MIN_EPSILON)**：`0.01` (1% 探索)
- **衰退回合數 (DECAY_EPISODES)**：`10000` 回合

```python
EPSILON = MAX_EPSILON - (MAX_EPSILON - MIN_EPSILON) × (episode / DECAY_EPISODES)
```

## 超參數設定

| 參數 | 值 | 說明 |
|------|-----|------|
| `ALPHA` | 0.1 | 學習率 |
| `GAMMA` | 0.99 | 折扣因子 |
| `NUM_EPISODES` | 20000 | 訓練回合數 |
| `MAX_STEPS_PER_EPISODE` | 100 | 每回合最大步數 |
| `EPSILON` (初始) | 1.0 | 探索率初始值 |
| `MIN_EPSILON` | 0.01 | 探索率最小值 |
| `DECAY_EPISODES` | 10000 | Epsilon 衰退的回合數 |

## 預期輸出

### 訓練過程

程式會每 2000 回合輸出一次當前的 Epsilon 值：

```
Episode: 2000, Epsilon: 0.8200
Episode: 4000, Epsilon: 0.6400
Episode: 6000, Epsilon: 0.4600
Episode: 8000, Epsilon: 0.2800
Episode: 10000, Epsilon: 0.1000
Episode: 12000, Epsilon: 0.0100
...
Training completed!
```

### 學習到的策略

訓練完成後，程式會輸出學習到的最佳策略 (每個狀態下應該選擇的動作方向)：

```
--- 學習到的最佳策略 (Policy) ---
 ↓  →  →  ↓
 ↓  ↓  →  ↓
 →  →  →  ↓
 →  →  →  G
```

**解讀：**
- 每個箭頭代表該狀態下 Q 值最高的動作
- `G` 代表目標狀態 (Goal)
- 理想情況下，策略應該指引 Agent 從任何狀態都能到達目標

### Q-Table 範例

程式也會輸出最終的 Q-Table (16×4 的陣列)，每一行代表一個狀態，每一列代表該動作的 Q 值。

## 實作細節

### 核心函式

1. **`step(state, action)`**
   - 實作環境的狀態轉移邏輯
   - 輸入：當前狀態和動作
   - 輸出：`(新狀態, 獎勵, 是否完成)`

2. **訓練迴圈**
   - 外層迴圈：遍歷所有訓練回合
   - 內層迴圈：每個回合中的步驟
   - 在每一步執行：
     1. 使用 Epsilon-Greedy 選擇動作
     2. 執行動作並獲得回饋
     3. 更新 Q-Table
     4. 更新狀態

### 資料結構

- **Q-Table**: `numpy.ndarray` (形狀: 16×4)
  - 行索引：狀態編號 (0-15)
  - 列索引：動作編號 (0-3)
  - 值：Q(狀態, 動作)

## 學習重點

這個實作展示了強化學習的核心概念：

1. ✅ **Agent-Environment 互動迴圈**：狀態 → 動作 → 獎勵 → 新狀態
2. ✅ **時間差分學習 (TD Learning)**：不需要等到回合結束就能更新價值
3. ✅ **探索與利用的權衡 (Exploration vs. Exploitation)**：Epsilon-Greedy 策略
4. ✅ **價值函數近似**：使用 Q-Table 儲存每個狀態-動作對的價值
5. ✅ **策略改進**：從學習到的 Q 值中提取最佳策略

## 限制與延伸

### 表格型方法的限制

- **維度詛咒 (Curse of Dimensionality)**：狀態空間稍大就無法使用
- **無法泛化**：每個狀態的學習是獨立的
- **記憶體需求**：需要儲存所有狀態-動作對

### 延伸閱讀

要突破這些限制，可以學習：
- **Deep Q-Network (DQN)**：使用神經網路取代 Q-Table → 參見 `3.DQN/`
- **Policy Gradient 方法**：直接學習策略而非價值函數 → 參見 `4.Policy_Gradients/`
- **Actor-Critic 方法**：結合價值與策略 → 參見 `5.PPO/`

## 參考資料

- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 6: Temporal-Difference Learning)
- Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards" (Ph.D. thesis)
