## <a name="installation-guide"></a>安裝指南 (Installation Guide)

[⬅️ 返回目錄](../README.md) | [下一章：模組一 - RL 的核心概念 ➡️](../1.Basics/README.md)

---

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
python 2.Q_Learning/2.1.Q_Learning/Q_Learning.py

# 執行 DQN 實作
python 3.DQN/3.1.Cart_Pole_DQN/cart_pole_dqn.py

# 執行 PPO 實作
python 5.PPO/5.1.pendulum/pendulum.py

# 執行 GRPO on MNIST 實作
python 7.GRPO/7.1.GRPO_MNIST/grpo_mnist.py

# GRPO on MaxText 實作需額外設定，請參考 7.GRPO/7.2.GRPO_Maxtext/README.md
python 7.GRPO/7.2.GRPO_Maxtext/reinforcement_learning_grpo.py
```

---

[⬅️ 返回目錄](../README.md) | [下一章：模組一 - RL 的核心概念 ➡️](../1.Basics/README.md)