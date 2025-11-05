## <a name="module-6-distributed-training-with-jaxmaxtext"></a>模組六：工程實作層 (JAX/Maxtext 分散式訓練)

[⬅️ 上一章：模組五 - PPO](../5.PPO/README.md) | [返回目錄](../README.md) | [下一章：模組七 - GRPO ➡️](../7.GRPO/README.md)

---

本模組回答了一個核心問題：「為什麼我們不能用一台電腦訓練 Llama 3.1 這種大型模型？以及 JAX/Maxtext 是如何解決這個問題的？」

### 6.1 規模的鴻溝：記憶體瓶頸 (The Memory Bottleneck)

**問題**：為什麼我們不能在「單一」的 GPU 或 TPU 核心上訓練一個 80 億 (8B) 參數的模型？

**答案**：**記憶體 (HBM)** 不夠用。

* **模型權重 (Model Weights)**：
    * `80 億` 參數 $\times$ `2 bytes/參數 (bfloat16)` = **16 GB**
* **梯度 (Gradients)**：
    * 我們的「學習引擎」(`nnx.value_and_grad`) 需要為**每一個**參數計算一個梯度。
    * 80 億 參數 = 80 億 梯度 = **另外 16 GB**
* **優化器狀態 (Optimizer State)**：
    * 我們使用的 `optax.adam` 是一個「自適應」優化器。
    * 它需要為**每一個**參數儲存**兩個**額外的「動量 (moments)」值 ( $m$ 和 $v$ )。
    * `80 億` 參數 $\times$ `2 個動量` $\times$ `2 bytes/值` = **另外 32 GB**

**總帳 (最低需求)：**
$$
16 \text{ GB (模型)} + 16 \text{ GB (梯度)} + 32 \text{ GB (Adam 狀態)} = \text{至少 64 GB}
$$

**結論**：這個 `64 GB` 的負載，**無法**放進一個只有 `32 GB` 或 `80 GB` HBM (高速頻寬記憶體) 的**單一**晶片中 (因為還需要額外空間進行計算)。

**解決方案**：我們**必須**把這個巨大的負載**「切分 (split)」**到**多個**晶片上才能執行。

---

### 6.2 關鍵概念 1：`Mesh` (硬體地圖)

在我們「切分」模型之前，我們必須先「描述」我們的硬體佈局。

* **`Mesh` (網格)**：這不是一個物理上的東西，而是 JAX 用來理解你的硬體叢集 (Cluster) 的一個「**邏輯地圖**」。
* **範例**：你有 `8` 個 TPU 核心。
    * **JAX 預設**：`[c1, c2, c3, c4, c5, c6, c7, c8]` (一個 1D 列表)
    * **你的 `Mesh`**：`[[c1, c2], [c3, c4], [c5, c6], [c7, c8]]` (一個 4x2 的 2D 網格)
* **`Mesh` 軸 (Axes)**：
    * `Maxtext` (例如 `config_ref.mesh_axes`) 會為這個 2D 網格的「維度」命名。
    * 軸 0 (長度為 4) $\rightarrow$ 命名為 `'data'`
    * 軸 1 (長度為 2) $\rightarrow$ 命名為 `'model'`
* **結論**：`Mesh` 建立了一個「硬體地圖」。我們現在有了一個 `(4, 2)` 的晶片網格，它有兩個「**邏輯物理軸**」：`data` 軸和 `model` 軸。

---

### 6.3 關鍵概念 2：兩種「平行」策略

有了「地圖 (`Mesh`)」，我們有兩種主要的方式來把「工作」分配下去：

**1. 資料平行 (Data Parallelism) - (「複製」策略)**
* **做法**：把**完整的 64GB** 模型**「複製 (copy)」**到**每一個**晶片上。
* **加速**：把「工作 (Batch)」切成 8 份，8 個晶片**同時**處理 8 份不同的資料。
* **優點**：速度快。
* **缺點**：**完全沒有解決記憶體問題**。

**2. 模型/張量平行 (Model/Tensor Parallelism) - (「切分」策略)**
* **做法**：把** 64GB** 的模型**「切成 (split)」** 8 塊，每塊 `8GB`。**每個**晶片**只**儲存它自己的那一塊 `8GB`。
* **加速**：晶片 1 計算完它的 `8GB`，把結果「傳遞 (communicate)」給晶片 2...
* **優點**：**完美解決了記憶體問題**。
* **缺點**：工程上極度複雜，需要大量晶片間通訊。

**JAX/Maxtext 的策略：兩全其美**
`Maxtext` 利用 `Mesh` 的**多維度** (例如 `('data', 'model')`) 來**同時**實現這兩種平行：

1.  JAX 看到 `'model'` 軸 (長度 2)，它執行「**模型平行**」：
    * 「OK，我把模型權重**切成 2 份**，沿著 `'model'` 軸存放。」
2.  JAX 看到 `'data'` 軸 (長度 4)，它執行「**資料平行**」：
    * 「OK，我把資料批次**切成 4 份**，沿著 `'data'` 軸分發。」

---

### 6.4 關鍵概念 3：`logical_axis_rules` (切分規則手冊)

**問題**：JAX 如何知道模型中「哪一層」該用「哪種方式」切分？

**答案**：`logical_axis_rules` (切分規則手冊)。

`logical_axis_rules` 是一本「**翻譯字典**」，它把**「模型內部」**的名稱，**翻譯**成**「硬體 (`Mesh`)」**上的名稱。

1.  **模型內部的「邏輯軸 (Logical Axes)」**：
    * `Maxtext` 在建立模型時，已為**每個**張量的**每個**維度取了「邏輯名稱」。
    * `(batch_size, sequence_length)` $\rightarrow$ `('batch', 'sequence')`
    * `(hidden_dim, mlp_dim)` $\rightarrow$ `('embed', 'mlp')`
    * `(vocab_size, hidden_dim)` $\rightarrow$ `('vocab', 'embed')`

2.  **規則手冊 (`config_policy.logical_axis_rules`)**：
    * 這是一個字典，你來定義「翻譯規則」。
    * `{ '邏N輯軸名稱': '物理 Mesh 軸名稱' }`

    ```python
    # 範例規則手冊 (Rulebook)
    rules = {
        'batch': 'data',    # 告訴 JAX：把 'batch' 維度，沿著 'data' 軸切 4 份
        'mlp':   'model',   # 告訴 JAX：把 'mlp' 維度，沿著 'model' 軸切 2 份
        'embed': None,      # 告訴 JAX：'embed' 維度 (hidden_dim)，「不」切分 (複製)
    }
    ```

3.  **`with nn_partitioning.axis_rules(...)` (自動建造者)**：
    * 這就是你程式碼中的 `with` 區塊。
    * **JAX 會自動執行**：
        1.  讀取你的「硬體地圖 (`Mesh`)」。
        2.  讀取你的「規則手冊 (`logical_axis_rules`)」。
        3.  當它建立 `nnx.Linear` (邏輯軸 `('embed', 'mlp')`) 時，它查詢手冊：
            * `'embed'` $\rightarrow$ `None` (不切)
            * `'mlp'` $\rightarrow$ `'model'` (沿 `model` 軸切 2 份)
        4.  **JAX 自動**將這個權重**切成 2 塊**，實現**「模型平行」**。
        5.  當它看到「資料 (Batch)」(邏輯軸 `('batch', 'sequence')`) 時，它查詢手冊：
            * `'batch'` $\rightarrow$ `'data'` (沿 `data` 軸切 4 份)
        6.  **JAX 自動**將資料**切成 4 份**，實現**「資料平行」**。

**總結**：你 (開發者) 只需要定義「硬體地圖 (`Mesh`)」和「切分規則 (`axis_rules`)」，`Maxtext` 和 `JAX` 就會自動幫你完成所有複雜的分散式訓練工作。

---

[⬅️ 上一章：模組五 - PPO](../5.PPO/README.md) | [返回目錄](../README.md) | [下一章：模組七 - GRPO ➡️](../7.GRPO/README.md)