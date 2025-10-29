# 強化學習 (RL) 完整學習筆記 (README.md)

這份文件涵蓋了從強化學習 (RL) 的核心概念，到 Deep Q-Networks (DQN)，再到 Proximal Policy Optimization (PPO) 的完整理論與實作筆記。

## 目錄 (Table of Contents)

0.  [安裝指南 (Installation Guide)](#installation-guide)
1.  [模組一：RL 的核心概念](#module-1-the-foundations-of-rl)
    * 1.1 核心 RL 迴圈 (Agent, Environment, S, A, R)
    * 1.2 馬可夫決策過程 (MDPs) 與馬可夫特性
    * 1.3 策略 (Policy) 與價值函數 (Value Function)
    * 1.4 探索 (Exploration) vs. 利用 (Exploitation)
2.  [模組二：表格型解法 (Q-Learning)](#module-2-tabular-methods-q-learning)
    * 2.1 Q-Function (Q 函數)
    * 2.2 時間差分學習 (TD Learning)
    * 2.3 Q-Learning 演算法 (Q-Table 更新)
    * 📁 [實作：`1_Q_Learning/Q_Learning.py`](1_Q_Learning/Q_Learning.py)
3.  [模組三：深度強化學習 (DQN)](#module-3-deep-q-networks-dqn)
    * 3.1 函數近似 (Function Approximation)
    * 3.2 DQN 關鍵技術 (Experience Replay & Target Network)
    * 3.3 DQN 實作 (Flax NNX)
    * 📁 [實作：`2_Cart_Pole_DQN/cart_pole_dqn.py`](2_Cart_Pole_DQN/cart_pole_dqn.py)
4.  [模組四：策略梯度 (Policy Gradients)](#module-4-policy-gradients-the-why)
    * 4.1 DQN 的局限性
    * 4.2 策略梯度 (REINFORCE) 演算法
    * 4.3 REINFORCE 的缺陷 (高變異數 & 信用分配)
5.  [模組五：Actor-Critic 與 PPO](#module-5-actor-critic--ppo-the-how)
    * 5.1 基線 (Baseline) 與優勢 (Advantage)
    * 5.2 Actor-Critic (演員-評論家) 架構
    * 5.3 PPO (Proximal Policy Optimization) 核心理論
    * 5.4 PPO 實作 (Flax NNX)
    * 📁 [實作：`3_Pendulum/pendulum.py`](3_Pendulum/pendulum.py)
6.  [專案實作總覽 (Project Implementations Summary)](#專案實作總覽-project-implementations-summary)

---

## <a name="installation-guide"></a>安裝指南 (Installation Guide)

本專案使用 Python 3.10.0，建議使用虛擬環境來管理依賴套件。

### 步驟 1：建立虛擬環境 (如果尚未建立)

```bash
python3 -m venv .venv
```

### 步驟 2：啟動虛擬環境

```bash
source .venv/bin/activate
```

### 步驟 3：安裝依賴套件

所有必要的 Python 套件已列在 `requirements.txt` 檔案中。執行以下指令一次性安裝：

```bash
pip install -r requirements.txt
```

**主要依賴套件包含：**
- **numpy**：陣列運算
- **jax + jaxlib**：高效能數值計算
- **flax**：JAX 神經網路框架 (NNX API)
- **optax**：最佳化演算法
- **gymnasium**：RL 環境 (CartPole-v1, Pendulum-v1)
- **tensorflow-probability[jax]**：機率分佈 (用於連續動作空間)

### 步驟 4：驗證安裝

安裝完成後，你可以執行以下指令來驗證環境設定：

```bash
# 執行 Q-Learning 實作
python 1_Q_Learning/Q_Learning.py

# 執行 DQN 實作
python 2_Cart_Pole_DQN/cart_pole_dqn.py

# 執行 PPO 實作
python 3_Pendulum/pendulum.py
```

---

## <a name="module-1-the-foundations-of-rl"></a>模組一：RL 的核心概念

本模組介紹了 RL 的基本「世界觀」和共同詞彙。

### 1.1 核心 RL 迴圈 (The Core Loop)

RL 的一切都基於「**智慧體 (Agent)**」和「**環境 (Environment)**」之間的互動迴圈。

1.  **智慧體 (Agent)**：學習者或決策者 (例如：遊戲角色)。
2.  **環境 (Environment)**：Agent 互動的外部世界 (例如：遊戲關卡)。
3.  **狀態 (State, $S$)**：環境在某一瞬間的快照 (例如：角色的 $(x, y)$ 座標)。
4.  **動作 (Action, $A$)**：Agent 根據 $S$ 能做出的選擇 (例如：「向左」)。
5.  **獎勵 (Reward, $R$)**：Agent 執行 $A$ 後，環境給予的「回饋訊號」 (例如：吃到金幣 `+10`，碰到敵人 `-50`)。

**Agent 的唯一目標**：最大化「**未來的累積總獎勵 (Cumulative Future Reward)**」。



### 1.2 馬可夫決策過程 (MDPs)

MDPs 是描述 RL 迴圈的數學框架。其核心假設是「**馬可夫特性 (Markov Property)**」。

> **馬可夫特性 (又稱「無記憶性」)**：
> 未來的狀態**只**取決於「現在的狀態」和「現在的動作」，而與「過去如何到達現在這個狀態」完全無關。

換句話說，**「現在的狀態 $S$」已經包含了所有做決策所需的歷史資訊**。
* **符合：** 圍棋 (棋盤的當下佈局就是一切)。
* **不符合：** 撲克 (你需要記憶對手「過去」的下注行為)。

### 1.3 策略 (Policy) 與價值函數 (Value Function)

這兩個概念描述了 Agent 的「大腦」和「目標」。

* **策略 (Policy, $\pi$)**：
    * Agent 的「行為準則」或「決策大腦」。
    * 它是一個函式，決定了在特定狀態 $S$ 下該採取哪個動作 $A$。
    * $\pi(A | S)$ = 在 $S$ 狀態下，執行 $A$ 動作的機率。
    * **我們的目標**：找到「**最佳策略 ($\pi^*$)**」，以獲得最高總獎勵。

* **價值函數 (Value Function, $V(s)$)**：
    * 一個「評價函式」，用來衡量「**一個狀態 $S$ 有多好**」。
    * $V(s)$ = 「從 $S$ 狀態出發，並遵循策略 $\pi$ 到底，**預期**能獲得的未來總獎勵。」
    * Agent 會利用 $V(s)$ 來改進 $\pi$ (例如：選擇那個能帶我走向「更高價值 $V(s')$」的動作 $A$)。

### 1.4 探索 (Exploration) vs. 利用 (Exploitation)

這是 Agent 學習時面臨的核心兩難。

* **利用 (Exploitation)**：
    * **定義**：根據**目前所知**的資訊，做出「最好」的選擇。
    * **例子**：去你最愛的那家餐廳 (你知道它 90 分)。
* **探索 (Exploration)**：
    * **定義**：**故意**嘗試一些「未知」的選擇，目的是為了「收集新資訊」。
    * **例子**：嘗試一家新餐廳 (可能是 10 分，也可能是 100 分)。

一個好的 Agent 必須在這兩者間取得平衡。**Epsilon-Greedy ($\epsilon$-Greedy)** 是最常見的策略：
* 有 $1-\epsilon$ 的機率 (例如 90%) 去「利用」。
* 有 $\epsilon$ 的機率 (例如 10%) 去「探索」。

---

## <a name="module-2-tabular-methods-q-learning"></a>模組二：表格型解法 (Q-Learning)

本模組介紹了第一個具體的 RL 演算法，適用於「狀態空間」很小 (例如 4x4 網格) 的問題。

### 2.1 Q-Function (Q 函數)

Q-Learning 引入了一個比 $V(s)$ 更強大的「**Q 函數**」，也叫「**Q 值**」。

* $V(s)$：「在狀態 $s$」有多好？
* $Q(s, a)$：「在狀態 $s$，**並且**執行動作 $a$」有多好？

$Q$ 函數更直接。Agent 在狀態 $s$ 時，不需要思考 $V(s')$，它只需要比較一下：
* $Q(s, \text{往左}) = 10$
* $Q(s, \text{往右}) = 50$
* ...然後選擇 $Q$ 值最高的動作 (往右)。

在表格型解法中，我們用一個「**Q-Table (Q 表格)**」 (例如 NumPy 陣列) 來儲存**每一個** $(s, a)$ 組合的 Q 值。

### 2.2 時間差分學習 (TD Learning)

Q-Learning 是一種「**時間差分 (Temporal-Difference, TD) 學習**」方法。

* **核心思想**：我們**不需要**等到遊戲結束，才回頭更新價值。我們在「**每走一步**」時，就「**用未來的現實，修正過去的預估**」。
* **例子**：你預估上班要 30 分鐘。5 分鐘後，你看到高速公路堵死了。你**立刻** (TD Learning) 更新了預估 (變成 2 小時)，而**不是** 2 小小時後才「學到」這件事。

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

**實作 (NumPy)**：
我們在 4x4 網格世界中，使用 NumPy 陣列 `q_table = np.zeros((16, 4))` 來實作了這個演算法。

📁 **完整實作程式碼**：[`1_Q_Learning/Q_Learning.py`](1_Q_Learning/Q_Learning.py)
📖 **詳細說明文件**：[`1_Q_Learning/README.md`](1_Q_Learning/README.md)

---

## <a name="module-3-deep-q-networks-dqn"></a>模組三：深度強化學習 (DQN)

本模組是從「表格型」方法到「深度學習」方法的關鍵飛躍。

### 3.1 函數近似 (Function Approximation)

**1. 「表格」的詛咒 (Curse of Dimensionality)**
* **問題**：Q-Table (模組二) 只能用在狀態空間**極小** (例如 16 個) 的問題上。
* **範例**：如果玩 Atari 遊戲，狀態是 $84 \times 84$ 像素的畫面。狀態的總數 (例如 $4^{84 \times 84}$) 遠超宇宙中的原子數。
* **結論**：我們**不可能**建立一個 Q-Table 來「儲存」所有 Q 值。

**2. 解決方案：函數近似 (Function Approximation)**
* **核心思想**：我們不要「**儲存 (store)**」所有的 Q 值，而是訓練一個「**估算器 (estimator)**」 (一個函數)，讓它來「**估算 (estimate)**」Q 值。
* **我們的估算器**：**深度神經網路 (Deep Neural Network)**。

**3. Deep Q-Network (DQN)**
* DQN 就是一個神經網路，它被訓練來**扮演「Q-Table」的角色**。
* **輸入 (Input)**：狀態 $S$ (例如：遊戲畫面或 CartPole 的 4 個數字)。
* **輸出 (Output)**：一個向量，代表**所有可能動作**的 Q 值。
    * `Q_Network(S)` $\rightarrow$ `[Q(S, a_1), Q(S, a_2), ...]`

**4. 函數近似的好處**
1.  **記憶體效率**：一個幾百萬參數的網路，可以為「無限」的狀態空間估算 Q 值。
2.  **泛化能力 (Generalization)**：
    * 在 Q-Table 中，狀態 `(1,1)` 和 `(1,2)` 是兩個**獨立**的條目。
    * 在神經網路中，`state (1,1)` 和 `state (1,2)` 是**非常相似 (similar)** 的輸入。
    * 網路在 `(1,1)` 學到的經驗 (例如「往右走很好」)，會**自動「泛化」**到 `(1,2)`，讓它猜到「在 (1,2) 往右走可能也不錯」。

### 3.2 DQN 關鍵技術 (Experience Replay & Target Network)

直接用 Q-Learning 的 TD 更新公式來訓練神經網路是**極度不穩定**的。DQN 引入了兩項關鍵技術 (穩定器) 來解決這個問題。

**1. 穩定器 #1：經驗回放 (Experience Replay)**
* **問題**：神經網路訓練最怕「**高度相關 (Correlated)**」的資料。如果我們用連續的遊戲經驗 `(s_t, s_{t+1}, s_{t+2}, ...)` 來訓練，網路會「過度擬合 (Overfit)」於當前的遊戲區域，並「**忘記**」它以前學到的東西。
* **解決方案**：建立一個「**回放緩衝區 (Replay Buffer)**」(一個 `deque`)。
    1.  **收集**：Agent 正常玩遊戲，把**每一**步的經驗 `(S, A, R, S')` 存進 Replay Buffer (例如儲存過去 10 萬步)。
    2.  **訓練**：當要訓練網路時，我們**不是**用「剛剛的經驗」，而是從 Replay Buffer 中**「隨機抽樣 (random sample)」**一小批 (mini-batch) (例如 64 筆) **不相關**的舊經驗。
* **好處**：打破了資料的相關性，讓訓練更穩定。

**2. 穩定器 #2：目標網路 (Target Network)**
* **問題**：「**移動的靶心 (Moving Target)**」問題。
* **理論**：在 Q-Learning 的更新中，我們用「一個網路」同時計算**「預測值」**和**「目標值」**。
    * $\text{TD Target} = R + \gamma \cdot \max Q_{\text{new}}(S', a')$
    * $\text{Loss} = (\text{TD Target} - Q_{\text{new}}(S, A))^2$
* **問題**：這就像你 (網路) 在瞄準一個靶心 (Target)，但靶心也是由你 (網路) 決定的。你一調整站姿 (更新權重)，靶心也跟著亂動，你永遠瞄不準。
* **解決方案**：使用**兩套**神經網路。
    1.  **線上網路 (Online Network, $Q_{\text{online}}$)**：
        * 這是我們**主要**在訓練的網路。
        * 它負責計算「**預測值**」$Q_{\text{online}}(S, A)$。
    2.  **目標網路 (Target Network, $Q_{\text{target}}$)**：
        * 這是 Online Network 的一個「**複製體**」，它的權重是**被凍結的**。
        * 它**只**負責計算「**目標值**」 $\text{TD Target} = R + \gamma \cdot \max Q_{\text{target}}(S', a')$。
* **運作**：
    1.  `Online Network` (射手) 去追逐 `Target Network` (**固定的靶心**)，訓練變得穩定。
    2.  每隔 `N` 步 (例如 1000 步)，我們才「同步」一次：把 `Online Network` 的**新權重**複製到 `Target Network` 上 (移動靶心)。

### 3.3 DQN 實作 (Flax NNX)

我們使用 Flax NNX 實作了 DQN Agent 來解決「CartPole-v1」問題。

* **`QNetwork(nnx.Module)`**：我們用 `nnx.Linear` 建立了一個 3 層的 MLP 作為函數近似器。
* **`ReplayBuffer(deque)`**：我們實作了 `add()` 和 `sample()` 方法。
* **`DQNAgent`**：
    * `__init__`：初始化了 `online_network` 和 `target_network`。
    * **關鍵 API (Flax NNX)**：
        * 使用 `nnx.Optimizer(model, optax.adam(...))` 來將 `optax` 優化器與模型**綁定**。
        * 使用 `nnx.state(online_model)` 來**提取**權重。
        * 使用 `nnx.update(target_model, online_state)` 來**複製**權重 (實現 `Target Network` 同步)。
* **`train_step` (訓練步驟)**：
    1.  從 `ReplayBuffer` 中 `sample()` 一個 `batch`。
    2.  **計算 Target**：`td_target = batch_rewards + GAMMA * jnp.max(self.target_network(batch_next_states), axis=1)`
    3.  **計算 Loss**：定義 `loss_fn`，計算「預測值」(來自 `self.online_network`) 和 `td_target` 之間的**均方誤差 (MSE)**。
    4.  **更新**：使用 `nnx.value_and_grad` 和 `self.optimizer.update(grads)` 來更新 `online_network`。

📁 **完整實作程式碼**：[`2_Cart_Pole_DQN/cart_pole_dqn.py`](2_Cart_Pole_DQN/cart_pole_dqn.py)
📖 **詳細說明文件**：[`2_Cart_Pole_DQN/README.md`](2_Cart_Pole_DQN/README.md)

---

## <a name="module-4-policy-gradients-the-why"></a>模組四：策略梯度 (Policy Gradients) - (PPO/GRPO 的基石)

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

**學習邏輯：REINFORCE (蒙地卡羅法)**
REINFORCE 是最基礎的策略梯度演算法。
1.  **玩一整局 (Episode)**：讓**當前**的策略網路 $\pi_{\theta}$ 從頭玩到尾。
2.  **計算總成績**：計算這一整局的「**折扣後總獎勵 ($R_{\text{total}}$)**」。
3.  **回顧 (Credit Assignment)**：
    * 回顧這一局中的**每一步** `(S, A)`。
    * **如果 $R_{\text{total}}$ 是「好」的 (例如 +200)**：我們就「**獎勵**」這一局中採取的**所有**動作，**提高** $\pi_{\theta}(A|S)$ 的機率。
    * **如果 $R_{\text{total}}$ 是「壞」的 (例如 +10)**：我們就「**懲罰**」這一局中採取的**所有**動作，**降低** $\pi_{\theta}(A|S)$ 的機率。

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

---

## <a name="module-5-actor-critic--ppo"></a>模組五：Actor-Critic 與 PPO (PPO/GRPO 的核心)

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

📁 **完整實作程式碼**：[`3_Pendulum/pendulum.py`](3_Pendulum/pendulum.py)
📖 **詳細說明文件**：[`3_Pendulum/README.md`](3_Pendulum/README.md)

---