# 強化學習 (RL) 完整學習筆記

這份文件涵蓋了從強化學習 (RL) 的核心概念，到價值為基礎方法 (Q-Learning、DQN)，再到策略為基礎方法 (Policy Gradients、PPO、GRPO) 的完整理論與實作筆記。內容包含從表格型方法到深度學習，從單機訓練到分散式訓練 (JAX/MaxText)，最終達到大型語言模型 (LLM) 的強化學習訓練 (GRPO on Llama 3.1 8B)。

## 目錄 (Table of Contents)

0.  [安裝指南 (Installation Guide)](0.Setup/README.md)
1.  [模組一：RL 的核心概念](1.Basics/README.md)
    * 1.1 核心 RL 迴圈 (Agent, Environment, S, A, R)
        * Episode vs Continuing Tasks
        * Reward vs Return vs Value (易混淆！)
        * 折扣因子 (Discount Factor, γ) 詳解
    * 1.2 馬可夫決策過程 (MDPs) 與馬可夫特性
        * 如何將非馬可夫問題轉換成馬可夫問題
        * 部分觀察 MDP (POMDP)
    * 1.3 策略 (Policy) 與價值函數 (Value Function)
        * 確定性策略 vs 隨機性策略
        * 價值函數的數學定義
    * 1.4 探索 (Exploration) vs. 利用 (Exploitation)
        * 常見探索策略對比 (ε-greedy, Softmax, Entropy Bonus)
    * 1.5 RL 方法的兩大家族 (預覽)
        * Value-Based vs Policy-Based 對比
2.  [模組二：表格型解法 (Q-Learning)](2.Q_Learning/README.md)
    * 2.1 Q-Function (Q 函數)
        * Q 函數 vs V 函數的數學關係
        * Q-Table 實作範例
    * 2.2 時間差分學習 (TD Learning)
        * Monte Carlo vs TD(0) vs TD(λ) 對比
        * Bootstrap (自舉) 的意義
    * 2.3 Q-Learning 演算法 (Q-Table 更新)
        * 完整更新公式與 Python 實作
    * 2.4 Off-Policy vs On-Policy (關鍵概念！)
        * Q-Learning vs SARSA 對比
        * 為什麼 Off-Policy 可以重用舊資料
    * 2.5 收斂性與超參數調整
        * 收斂保證 (Robbins-Monro 條件)
        * 超參數表 (α, γ, ε)
        * 常見問題與調試技巧
    * 📁 [實作：`2.Q_Learning/2.1.Q_Learning/Q_Learning.py`](2.Q_Learning/2.1.Q_Learning/Q_Learning.py)
3.  [模組三：深度強化學習 (DQN)](3.DQN/README.md)
    * 3.1 函數近似 (Function Approximation)
        * 維度詛咒問題
        * 神經網路的泛化能力
    * 3.2 為什麼 DQN 無法處理連續動作？
        * 數學原因與常見錯誤嘗試
    * 3.3 DQN 關鍵技術 (Experience Replay & Target Network)
        * 為什麼直接套用 Q-Learning 會失敗
        * Hard Update vs Soft Update (Polyak)
    * 3.4 DQN 的過估計問題 (Overestimation Bias)
        * 數學證明與 Double DQN 解決方案
    * 3.5 DQN 實作 (Flax NNX)
    * 3.6 ReplayBuffer 實作細節
        * Buffer 未滿時的採樣策略
        * 優先經驗回放 (PER)
    * 3.7 DQN 家族演算法進化圖
        * Rainbow DQN (整合 6 項技術)
    * 3.8 DQN 超參數與調試技巧
        * CartPole vs Atari 超參數對比
        * 常見問題解決方案
    * 📁 [實作：`3.DQN/3.1.Cart_Pole_DQN/cart_pole_dqn.py`](3.DQN/3.1.Cart_Pole_DQN/cart_pole_dqn.py)
4.  [模組四：策略梯度 (Policy Gradients)](4.Policy_Gradients/README.md)
    * 4.1 DQN 的局限性
        * 無法處理連續動作的數學原因
    * 4.2 策略梯度 (REINFORCE) 演算法
        * 策略梯度定理 (Policy Gradient Theorem)
        * 為什麼可以處理連續動作
        * 完整演算法與 Python 實作框架
    * 4.3 REINFORCE 的致命缺陷
        * 高變異數 & 信用分配問題
        * 缺乏「標準」(No Baseline) 問題
    * 4.4 從 REINFORCE 到 Actor-Critic 的演進
        * 引入 Baseline 降低變異數
        * 用 TD 取代 MC
    * 4.5 策略梯度方法對比表
        * REINFORCE → Actor-Critic → A2C → PPO → GRPO
    * 4.6 關鍵洞察：On-Policy 的必然性
        * 重要性採樣 (Importance Sampling)
5.  [模組五：Actor-Critic 與 PPO](5.PPO/README.md)
    * 5.1 基線 (Baseline) 與優勢 (Advantage)
    * 5.2 Actor-Critic (演員-評論家) 架構
    * 5.3 PPO (Proximal Policy Optimization) 核心理論
        * PPO-Clip 安全鎖
        * KL 散度懲罰 (Reference Model)
    * 5.4 PPO 實作 (Flax NNX)
        * Actor, Critic, RolloutBuffer 架構
    * 5.4.1 GAE 詳細解釋 (最易混淆的部分！)
        * 完整 GAE 推導
        * λ 參數的作用 (λ=0 vs λ=1 vs λ=0.95)
        * 實作範例與易錯點
        * 為什麼要標準化 advantages
    * 5.5 PPO vs DQN 完整對比
        * 為什麼 LLM 用 PPO 而非 DQN
    * 5.6 PPO 超參數調整指南
        * 關鍵超參數表
        * 調試流程與常見錯誤解決
    * 5.7 為什麼需要「舊」log_prob？
        * 重要性採樣 (Importance Sampling) 詳解
        * PPO 的創新：Clip 防止高變異數
    * 📁 [實作：`5.PPO/5.1.pendulum/pendulum.py`](5.PPO/5.1.pendulum/pendulum.py)
6.  [模組六：工程實作層 (JAX/Maxtext 分散式訓練)](6.Jax&Maxtext/README.md)
    * 6.1 規模的鴻溝：記憶體瓶頸 (The Memory Bottleneck)
        * 為什麼無法在單一 GPU/TPU 訓練 8B 模型
        * 模型權重 + 梯度 + 優化器狀態的記憶體計算
    * 6.2 關鍵概念 1：`Mesh` (硬體地圖)
        * 如何將物理硬體組織成邏輯網格
    * 6.3 關鍵概念 2：兩種「平行」策略
        * 資料平行 (Data Parallelism)
        * 模型/張量平行 (Model/Tensor Parallelism)
        * JAX/Maxtext 如何兩全其美
    * 6.4 關鍵概念 3：`logical_axis_rules` (切分規則手冊)
        * 邏輯軸 → 物理軸的翻譯機制
7.  [模組七：GRPO (Group Relative Policy Optimization)](7.GRPO/README.md)
    * 7.1 GRPO 的核心取捨 (Trade-off)
        * PPO vs GRPO：記憶體 vs 計算的權衡
        * 群組 (Group) 的詳細定義 (最易混淆！)
        * 相對優勢計算：個體表現 vs 群組平均
    * 7.2 專案實作：GRPO on MNIST
        * 將分類問題 RL 化
        * 無 Critic 網路的實作
        * 群組基線計算
    * 7.3 GRPO vs PPO 完整對比
        * 為什麼 LLM 訓練偏好 GRPO
        * 記憶體需求計算 (Llama 3.1 8B 範例)
    * 7.4 Group Size (G) 的選擇
        * G 太小/太大的問題
        * 不同場景的推薦值
        * 調整策略
    * 7.5 GRPO 在 LLM RLHF 的完整流程
        * 與標準 PPO RLHF 的差異對比
        * 完整代碼示例
    * 7.6 易混淆點總結
        * 3 個常見誤解 + 正確理解
    * 📁 [實作：`7.GRPO/7.1.GRPO_MNIST/grpo_mnist.py`](7.GRPO/7.1.GRPO_MNIST/grpo_mnist.py)
    * 📁 [實作：`7.GRPO/7.2.GRPO_Maxtext/grpo_llama3_1_8b_demo_pw.py`](7.GRPO/7.2.GRPO_Maxtext/grpo_llama3_1_8b_demo_pw.py)
---