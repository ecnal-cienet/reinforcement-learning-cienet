# Group Relative Policy Optimization (GRPO) with MaxText - LLM 訓練於 GSM8K

> **English Version**: [README.md](README.md)

## 概述

這是一個將 **Group Relative Policy Optimization (GRPO)** 演算法應用於訓練 **Llama 3.1 8B-Instruct 模型**處理 **GSM8K 數學推理基準測試**的實作。這代表了本學習資料庫的學習進程巔峰，將 GRPO 應用於真實世界的大型語言模型場景。

**核心成就：** 本實作展示了生產規模的 GRPO，訓練一個 80 億參數的 LLM 以提升數學推理能力，使用強化學習技術。

**與前面專案的關係：**
- `2.Q_Learning/`: 價值為基礎 - 表格型 Q-Learning
- `3.DQN/`: 價值為基礎 - 深度 Q 網路與函數逼近
- `5.PPO/`: 策略為基礎 - PPO 與 Actor-Critic
- `7.GRPO/7.1.GRPO_MNIST/`: 策略為基礎 - GRPO 於簡單分類
- `7.GRPO/7.2.GRPO_Maxtext/` **(本專案)**: 策略為基礎 - **生產級 GRPO 於 8B LLM**

## 為什麼使用 GRPO 訓練 LLM？

### LLM 對齊的挑戰

大型語言模型需要透過**基於人類反饋的強化學習 (Reinforcement Learning from Human Feedback, RLHF)** 來與人類偏好對齊。傳統的 PPO 面臨嚴重挑戰：

**PPO 在 8B 模型上的記憶體需求：**
- **Actor 網路** (8B 參數): ~64 GB (權重 + 梯度 + 優化器狀態)
- **Critic 網路** (8B 參數): ~64 GB (權重 + 梯度 + 優化器狀態)
- **總計**: 128 GB (僅訓練狀態)

即使在高階硬體上，這也是令人望而卻步的昂貴。

### GRPO 針對 LLM 的解決方案

**核心創新：** 完全移除 Critic 網路，用從多個回應樣本計算的群組相對基線取代。

**取捨：**
- ✅ **記憶體效率**: 省下 50% 記憶體 (無 Critic 網路)
- ✅ **簡化訓練**: 不需要訓練和同步 Critic
- ❌ **推理成本**: 必須為每個提示生成 G 個回應 (而非 1 個)
- ❌ **樣本效率**: 在唯一樣本方面低於 PPO

**為什麼這對 LLM 有效：**
- LLM 推理成本 (生成 4-8 個回應) << 訓練 8B 參數 Critic 的成本
- 現代推理引擎 (vLLM) 使多次生成相對便宜
- 群組比較為推理任務提供自然基線

## 環境說明

### GSM8K 數學推理任務

**資料集：** Grade School Math 8K (GSM8K) - 需要多步驟推理的小學程度數學文字題資料集。

**範例問題：**
```
問題: "Janet 的鴨子每天下 16 顆蛋。她每天早上吃三顆當早餐，
每天用四顆幫朋友烤馬芬。她每天在農夫市集以每顆新鮮鴨蛋 $2 的價格
賣掉剩餘的蛋。她每天在農夫市集賺多少美元？"

答案: 18
```

### 作為 RL 問題的任務

**RL 框架：**
```
State (提示) → 策略生成回應 → 獎勵模型評分 → 更新策略
```

**組成部分：**
- **State**: 使用者提示 (數學文字題)
- **Action**: 生成的文字序列 (推理 + 答案)
- **Reward**: 基於正確性和格式的自訂獎勵函數
- **Episode**: 單次 (每個提示都是獨立的)

### 提示模板

模型被指示使用結構化推理：

```python
SYSTEM_PROMPT = """你被給予一個問題。思考這個問題並提供你的推理。
將推理放在 <reasoning> 和 </reasoning> 之間。
然後，在 <answer> 和 </answer> 之間提供最終答案 (即只有一個數值)。"""
```

**預期輸出格式：**
```
<reasoning>
Janet 每天得到 16 顆蛋。
她吃 3 顆當早餐: 16 - 3 = 13 顆蛋剩餘。
她用 4 顆做馬芬: 13 - 4 = 9 顆蛋剩餘。
她以每顆 $2 賣掉這些: 9 × $2 = $18
</reasoning>
<answer>18</answer>
```

### 獎勵函數

實作使用多組成的獎勵系統：

1. **格式匹配獎勵**
   - 精確格式匹配: +3.0 分
   - 空白字元格式匹配: +1.5 分
   - 部分格式匹配: 每個標籤 +0.5 分
   - 錯誤格式懲罰: -0.5 分

2. **答案準確性獎勵**
   - 精確答案: +3.0 分
   - 答案在 10% 比率內: +0.5 分
   - 答案在 20% 比率內: +0.25 分
   - 錯誤答案懲罰: -1.0 分

3. **數字提取獎勵**
   - 正確提取的數字匹配: +1.5 分

這個多面向的獎勵鼓勵正確答案和適當的推理格式。

## 如何執行

### 前置需求

本實作需要 MaxText 框架和相關依賴。

#### 1. 安裝 MaxText

遵循官方 MaxText 安裝指南：
```bash
# 參見: https://maxtext.readthedocs.io/en/latest/guides/install_maxtext.html
```

#### 2. 安裝額外依賴

```bash
source .venv/bin/activate
pip install -r requirements.txt

# 本模組的額外需求:
pip install transformers tensorflow-datasets grain orbax-checkpoint
pip install tunix  # GRPO 框架
```

#### 3. 準備模型檢查點

將 Llama 3.1 8B-Instruct 轉換為 MaxText 格式：
```bash
# 使用 MaxText 檢查點轉換腳本
# 參見: maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
```

實作預期的掃描檢查點位於：
```python
MODEL_CHECKPOINT_PATH = "gs://your-bucket/llama3.1-8b-Instruct/scanned-pathways/0/items"
```

#### 4. 設定雲端儲存

更新腳本中的以下路徑：
- `CKPT_DIR`: 儲存訓練檢查點的位置
- `PROFILE_DIR`: 儲存分析資料的位置
- `LOG_DIR`: 儲存 TensorBoard 日誌的位置

### 執行程式

**硬體需求：** TPU VM (推薦 v5p-8、v6e-8 或 v5p-64)

```bash
# 從資料庫根目錄
python 7.GRPO/7.2.GRPO_Maxtext/grpo_llama3_1_8b_demo_pw.py
```

**注意：** 此腳本設計在具有 Pathways 整合的 Google Cloud TPU VM 上執行。

### 監控訓練

在 TensorBoard 中查看訓練指標：
```bash
tensorboard --logdir ~/content/tensorboard/grpo/logs_llama3/ --port=8086
```

## 演算法核心

### LLM 情境中的 GRPO

GRPO 將 PPO 演算法調整用於大型語言模型訓練，透過用群組相對基線取代 Critic 網路。

#### 與 MNIST GRPO 的主要差異

| 方面 | GRPO-MNIST | GRPO-MaxText (LLM) |
|------|-----------|-------------------|
| **問題類型** | 單步驟分類 | 多步驟序列生成 |
| **State** | 單一圖片 (784-dim) | 文字提示 (可變長度) |
| **Action** | 單一數字 (0-9) | 標記序列 (最多 768 個標記) |
| **模型大小** | ~100K 參數 | 80 億參數 |
| **訓練框架** | 純 JAX/Flax | MaxText + Tunix |
| **推理引擎** | 直接模型呼叫 | vLLM (優化生成) |
| **分散式** | 單一裝置 | 多裝置 (TP/PP) |

### LLM 的 GRPO 訓練迴圈

```python
for batch in dataset:  # 每個批次包含提示
    # ========== 階段 1: Rollout (生成多個回應) ==========
    # 為每個提示生成 G 個回應
    prompts = batch["prompts"]  # 例如, ["數學問題 1", "數學問題 2", ...]

    responses = []
    log_probs_old = []
    for g in range(NUM_GENERATIONS):  # G = 2 在本實作中
        # 使用 vLLM 進行高效生成
        resp = vllm_generate(prompts, temperature=0.9, top_k=50)
        log_prob = compute_log_probs(prompts, resp)
        responses.append(resp)
        log_probs_old.append(log_prob)

    # ========== 階段 2: 計算獎勵 ==========
    # 使用獎勵函數為每個回應評分
    rewards = []
    for resp in responses:
        reward = 0.0
        reward += match_format_exactly(resp)       # 格式正確 +3.0
        reward += match_format_approximately(resp) # 部分得分
        reward += check_answer(resp, true_answer)  # 答案正確 +3.0
        reward += check_numbers(resp, true_answer) # 提取數字
        rewards.append(reward)

    # ========== 階段 3: 計算群組相對優勢 ==========
    # 這是 GRPO 的關鍵創新 - 不需要 Critic！
    group_baseline = mean(rewards)  # G 次生成的平均獎勵
    advantages = rewards - group_baseline

    # 正規化優勢
    advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)

    # ========== 階段 4: 策略更新 (PPO-Clip) ==========
    # 使用 PPO-Clip 目標與 KL 懲罰更新策略
    loss = compute_grpo_loss(
        prompts, responses, log_probs_old, advantages,
        beta=BETA,      # KL 懲罰係數 (0.08)
        epsilon=EPSILON  # 裁剪參數 (0.2)
    )

    # 更新模型參數
    optimizer.step(loss)
```

### GRPO 損失函數

```python
def compute_grpo_loss(prompts, responses, log_probs_old, advantages, beta, epsilon):
    # (1) 用更新的策略計算新的對數機率
    log_probs_new = policy_model.compute_log_probs(prompts, responses)

    # (2) 計算策略比率
    ratio = exp(log_probs_new - log_probs_old)

    # (3) PPO-Clip 目標
    loss_unclipped = advantages * ratio
    ratio_clipped = clip(ratio, 1 - epsilon, 1 + epsilon)
    loss_clipped = advantages * ratio_clipped
    ppo_loss = -mean(minimum(loss_unclipped, loss_clipped))

    # (4) KL 散度懲罰 (保持策略接近參考)
    kl_div = compute_kl(policy_model, reference_model, prompts)

    # (5) 組合損失
    total_loss = ppo_loss + beta * kl_div
    return total_loss
```

**關鍵組成：**
- **PPO-Clip**: 與標準 PPO 相同，防止大的策略更新
- **KL 懲罰**: 防止策略偏離參考模型太遠
- **無 Critic**: 優勢從群組統計計算，而非學習的價值函數

## 技術架構

### 基礎設施堆疊

```
┌─────────────────────────────────────────────┐
│         應用層                              │
│  grpo_llama3_1_8b_demo_pw.py               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         GRPO 訓練層                         │
│  - GrpoLearner (Tunix)                     │
│  - RLCluster (管理 Actor/Ref/Rollout)     │
│  - 獎勵函數                                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         模型層                              │
│  - Actor: Llama 3.1 8B (可訓練)           │
│  - Reference: Llama 3.1 8B (凍結)         │
│  - MaxText 模型實作                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         推理引擎                            │
│  - vLLM (高效批次生成)                     │
│  - JAX 後端用於 TPU                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         硬體層                              │
│  - TPU VMs (v5p-8, v6e-8, v5p-64)         │
│  - Pathways 用於分散式執行                 │
└─────────────────────────────────────────────┘
```

### 模型架構：Llama 3.1 8B

**Transformer 規格：**
- **參數**: 80 億
- **層數**: 32 個 transformer 區塊
- **隱藏大小**: 4096
- **注意力頭**: 32
- **上下文長度**: 128K 標記 (雖然本實作使用較短的上下文)

**MaxText 整合：**
實作使用 MaxText 的 Flax NNX 適配器 (`TunixMaxTextAdapter`)，將 MaxText 模型包裝以與 Tunix 的 GRPO 訓練框架相容。

```python
# 模型初始化
model, mesh = model_creation_utils.create_nnx_model(config, devices)
tunix_model = TunixMaxTextAdapter(base_model=model)
tunix_model.config = llama3_lib.ModelConfig.llama3_1_8b()
```

### 分散式訓練設定

**裝置分配 (多 VM 設定)：**
使用多個 TPU VM 時，實作在以下之間分割裝置：
- **訓練裝置** (50%): 執行策略訓練
- **取樣裝置** (50%): 執行 vLLM 生成

```python
num_trainer_devices = int(num_devices * TRAINER_DEVICES_FRACTION)  # 50%
num_sampler_devices = int(num_devices * SAMPLER_DEVICES_FRACTION)  # 50%
```

**Mesh 設定：**
```python
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: actor_mesh,      # 訓練 mesh
        rl_cluster_lib.Role.REFERENCE: reference_mesh,  # 參考模型 mesh
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,  # vLLM 生成 mesh
    }
)
```

### vLLM 整合

**為什麼使用 vLLM？**
- 高度優化用於 TPU/GPU 上的批次推理
- 支援高效 KV-cache 管理
- 可以並行生成多個序列
- 比原始生成迴圈快得多

**vLLM 設定：**
```python
rollout_config = base_rollout.RolloutConfig(
    max_tokens_to_generate=768,
    max_prompt_length=256,
    kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
    temperature=0.9,      # 高溫度用於多樣訓練回應
    top_p=1.0,
    top_k=50,
    rollout_vllm_model_version="meta-llama/Meta-Llama-3.1-8B-Instruct",
    rollout_vllm_hbm_utilization=0.2,  # 記憶體分配
    rollout_vllm_tpu_backend_type="jax",
)
```

## 超參數

### 生成參數 (訓練期間)

| 參數 | 值 | 描述 |
|-----|---|-----|
| `MAX_PROMPT_LENGTH` | 256 | 最大提示標記數 |
| `TOTAL_GENERATION_STEPS` | 768 | 最大生成標記數 |
| `TEMPERATURE` | 0.9 | 取樣溫度 (高以保持多樣性) |
| `TOP_P` | 1.0 | Nucleus 取樣閾值 |
| `TOP_K` | 50 | Top-k 取樣 |
| `NUM_GENERATIONS` | 2 | 每個提示的回應數 (G) |

**注意：** 高溫度 (0.9) 在訓練期間至關重要，以生成多樣化的回應進行有意義的群組比較。

### GRPO 訓練參數

| 參數 | 值 | 描述 |
|-----|---|-----|
| `NUM_ITERATIONS` | 1 | 小批次迭代 (論文中的 μ) |
| `BETA` | 0.08 | KL 散度懲罰係數 |
| `EPSILON` | 0.2 | PPO 裁剪參數 |
| `BATCH_SIZE` | 1 | 每批次提示數 |
| `NUM_BATCHES` | 3,738 | 總訓練批次數 |
| `NUM_EPOCHS` | 1 | 資料集通過次數 |

### 優化參數

| 參數 | 值 | 描述 |
|-----|---|-----|
| `LEARNING_RATE` | 3e-6 | 峰值學習率 |
| `B1` | 0.9 | Adam beta1 |
| `B2` | 0.99 | Adam beta2 |
| `WEIGHT_DECAY` | 0.1 | L2 正規化 |
| `WARMUP_STEPS` | 總數的 10% | LR 預熱步驟 |
| `MAX_GRAD_NORM` | 0.1 | 梯度裁剪閾值 |

**注意：** 非常小的學習率 (3e-6) 和積極的梯度裁剪 (0.1) 對於防止 KL 散度爆炸至關重要。

### 檢查點

| 參數 | 值 | 描述 |
|-----|---|-----|
| `SAVE_INTERVAL_STEPS` | 500 | 檢查點之間的步驟數 |
| `MAX_TO_KEEP` | 4 | 保留的檢查點數量 |

## 核心實作細節

### 1. 資料預處理

```python
def get_dataset(data_dir, split="train") -> grain.MapDataset:
    # 透過 TensorFlow Datasets 下載 GSM8K
    data = tfds.data_source("gsm8k", split=split, data_dir=data_dir)

    loaded_dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=SEED)
        .map(lambda x: {
            # 應用對話模板
            "prompts": model_tokenizer.apply_chat_template([{
                "role": "user",
                "content": TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=x["question"].decode("utf-8"),
                )
            }], tokenize=False, add_generation_prompt=True),

            # 提取真實答案
            "question": x["question"].decode("utf-8"),
            "answer": extract_hash_answer(x["answer"].decode("utf-8")),
        })
    )
    return loaded_dataset
```

**關鍵步驟：**
1. 從 TensorFlow Datasets 載入 GSM8K
2. 應用 Llama 3.1 對話模板
3. 從解決方案格式中提取數值答案
4. 為訓練建立批次

### 2. 獎勵函數實作

實作使用四個互補的獎勵函數依序呼叫：

```python
grpo_trainer = GrpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,       # 精確格式匹配
        match_format_approximately, # 部分格式匹配
        check_answer,               # 答案正確性
        check_numbers,              # 數字提取
    ],
    grpo_config=grpo_config,
)
```

**獎勵計算範例：**
```python
# 給定一個回應:
response = """
<reasoning>
16 - 3 = 13 顆蛋。13 - 4 = 9 顆蛋。9 × $2 = $18
</reasoning>
<answer>18</answer>
"""

# 獎勵累積:
reward = 0.0
reward += 3.0   # match_format_exactly (完美格式)
reward += 2.0   # match_format_approximately (所有 4 個標籤存在)
reward += 3.0   # check_answer (正確答案)
reward += 1.5   # check_numbers (提取的數字匹配)
# 總計: 9.5 分
```

### 3. 評估函數

```python
def evaluate(dataset, rl_cluster, temperature=0.7, top_k=50, top_p=0.95, num_passes=1):
    """
    在測試集上評估模型。

    指標:
    - 答案準確率: 精確匹配百分比
    - 部分準確率: 在 10% 比率內
    - 格式準確率: 正確格式百分比
    """
    corr = 0
    partially_corr = 0
    corr_format = 0
    total = 0

    for batch in tqdm(dataset):
        prompts = batch["prompts"]
        answers = batch["answer"]

        # 生成回應
        responses = generate_responses(prompts, rl_cluster, num_passes, temperature, top_k, top_p)

        # 為回應評分
        for response, answer in zip(responses, answers):
            if check_exact_match(response, answer):
                corr += 1
            if check_partial_match(response, answer):
                partially_corr += 1
            if check_format(response):
                corr_format += 1
            total += 1

    return corr / total * 100, partially_corr / total * 100, corr_format / total * 100
```

### 4. 訓練迴圈

```python
# 設定 GRPO 訓練器
grpo_trainer = GrpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=[...],
    grpo_config=grpo_config,
)

# 訓練前評估
accuracy_pre = evaluate(test_dataset, rl_cluster, **GENERATION_CONFIGS["greedy"])
print(f"訓練前: {accuracy_pre=}%")

# 訓練！
with mesh, nn_partitioning.axis_rules(config_policy.logical_axis_rules):
    grpo_trainer.train(train_dataset)

# 訓練後評估
accuracy_post = evaluate(test_dataset, rl_cluster, **GENERATION_CONFIGS["greedy"])
print(f"訓練後: {accuracy_post=}%")
```

## 預期行為

### 訓練過程

```
JAX 裝置: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]

載入 Llama 3.1 8B 模型...
模型初始化成功
模型 mesh 形狀: {'data': 1, 'fsdp': 8, 'tensor': 1}

初始化 vLLM 取樣器...
vLLM 在 rollout mesh 上就緒

在 GSM8K 上開始 GRPO 訓練...

訓練前評估:
  答案準確率: 42.3%
  部分準確率: 48.7%
  格式準確率: 31.2%

步驟 500/3738 | Loss: 2.341 | KL: 0.023 | 平均獎勵: 4.2
步驟 1000/3738 | Loss: 1.987 | KL: 0.031 | 平均獎勵: 5.1
步驟 1500/3738 | Loss: 1.743 | KL: 0.028 | 平均獎勵: 5.8
...
步驟 3738/3738 | Loss: 1.201 | KL: 0.025 | 平均獎勵: 7.3

訓練後評估:
  答案準確率: 58.9%
  部分準確率: 64.2%
  格式準確率: 87.5%

訓練完成！
檢查點已儲存至: gs://your-bucket/grpo/.../ckpts_llama3/
```

### 效能指標

**預期改進：**
- **答案準確率**: 比基礎模型提升 +10-20%
- **格式準確率**: 提升 +40-60% (模型學習結構化輸出)
- **推理品質**: 更多逐步解釋

**訓練時間：**
- **v5p-8**: 完整訓練約 ~6-8 小時
- **v5p-64**: 完整訓練約 ~1-2 小時

### 推理範例

**訓練前：**
```
問題: Janet 的鴨子每天下 16 顆蛋...

回應:
所以 Janet 有鴨子而且牠們下蛋。答案可能與賣蛋有關。
我認為是 20 美元。
```

**訓練後：**
```
問題: Janet 的鴨子每天下 16 顆蛋...

回應:
<reasoning>
Janet 每天從 16 顆蛋開始。
她吃 3 顆當早餐: 16 - 3 = 13 顆蛋剩餘。
她用 4 顆做馬芬: 13 - 4 = 9 顆蛋剩餘。
她以每顆 $2 賣掉這 9 顆蛋: 9 × $2 = $18。
</reasoning>
<answer>18</answer>
```

## 比較：從 MNIST 到 LLM

| 方面 | GRPO-MNIST (模組 7.1) | GRPO-MaxText (模組 7.2) |
|------|---------------------|----------------------|
| **模型大小** | ~100K 參數 (3 層 MLP) | 8B 參數 (32 層 Transformer) |
| **任務** | 數字分類 | 數學推理 |
| **輸入** | 784-dim 向量 | 可變長度文字 |
| **輸出** | 單一動作 (0-9) | 標記序列 (最多 768) |
| **回合** | 單步驟 | 多步驟生成 |
| **獎勵** | 二元 (0/1) | 複合 (0-10+ 分) |
| **訓練時間** | 分鐘 | 小時 |
| **框架** | 純 JAX/Flax | MaxText + Tunix + vLLM |
| **硬體** | 單一 CPU/GPU | 多晶片 TPU |
| **群組大小 (G)** | 1024 | 2 (受計算限制) |

**關鍵洞察：** 核心 GRPO 演算法 (群組相對基線) 保持相同，但規模和基礎設施需求有很大差異。

## 進階主題

### 1. KL 散度監控

監控 KL 散度以確保策略不會偏離太遠：

```python
# 訓練期間，追蹤 KL
kl_div = compute_kl(policy_model, reference_model, prompts)

# 如果 KL > 閾值，增加 BETA
if kl_div > 0.1:
    BETA *= 1.5  # 增加懲罰
```

### 2. 自適應溫度

根據訓練進度調整生成溫度：

```python
# 早期訓練: 高溫度用於探索
TEMPERATURE = 0.9

# 後期訓練: 隨策略穩定降低溫度
TEMPERATURE = max(0.7, 0.9 - epoch * 0.02)
```

### 3. LoRA 整合

為了更高的記憶體效率，整合 LoRA (低秩適應)：

```python
# 只訓練低秩適配器矩陣
# 基礎模型保持凍結
# 進一步減少記憶體需求
```

### 4. 多任務訓練

擴展到多個推理資料集：

```python
# 結合 GSM8K 與其他資料集
datasets = [gsm8k_dataset, math_dataset, code_dataset]
for batch in interleave(datasets):
    train_step(batch)
```

## 生產考量

### 記憶體管理

**關鍵參數：**
```python
HBM_UTILIZATION_VLLM = 0.2  # vLLM 記憶體分配
TRAINER_DEVICES_FRACTION = 0.5  # 裝置分割
```

根據可用 HBM 調整：
- v5p-8: 每晶片 16 GB → 使用保守設定
- v6e-8: 每晶片 32 GB → 可以增加批次大小

### 檢查點策略

```python
# 頻繁儲存檢查點
SAVE_INTERVAL_STEPS = 500

# 保留最近的檢查點用於恢復
MAX_TO_KEEP = 4

# 使用非同步檢查點以提高效率
async_checkpointing = "true"
```

### 監控

**需追蹤的關鍵指標：**
1. **Loss**: 應穩定下降
2. **KL 散度**: 應保持 < 0.1
3. **平均獎勵**: 應增加
4. **準確率**: 每 N 步評估

**TensorBoard 視覺化：**
```bash
tensorboard --logdir ~/content/tensorboard/grpo/logs_llama3/ --port=8086
```

## 故障排除

### 常見問題

**1. OOM (記憶體不足)**
```
解決方案:
- 減少 BATCH_SIZE
- 減少 HBM_UTILIZATION_VLLM
- 減少 TOTAL_GENERATION_STEPS
- 啟用梯度檢查點
```

**2. KL 散度爆炸**
```
解決方案:
- 增加 BETA (KL 懲罰)
- 減少 LEARNING_RATE
- 增加 MAX_GRAD_NORM 裁剪
```

**3. vLLM 初始化失敗**
```
解決方案:
- 確保正確的模型路徑
- 檢查裝置分配 (vLLM 有足夠的裝置)
- 驗證 vLLM 與 JAX 後端的相容性
```

**4. 生成緩慢**
```
解決方案:
- 增加 HBM_UTILIZATION_VLLM (如果記憶體允許)
- 如果可能減少 NUM_GENERATIONS
- 使用更大的 TPU 配置
```

## 參考資料

### 論文
- DeepSeek-R1 (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms" ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback" (InstructGPT)
- Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems" (GSM8K Dataset)

### 框架與函式庫
- [MaxText 文件](https://maxtext.readthedocs.io/)
- [Tunix GRPO 實作](https://github.com/google/tunix)
- [vLLM: 快速 LLM 推理](https://github.com/vllm-project/vllm)
- [GSM8K 資料集](https://huggingface.co/datasets/gsm8k)
- [Llama 3.1 模型卡](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### 相關工作
- Anthropic. "Constitutional AI: Harmlessness from AI Feedback"
- OpenAI. "Learning to Summarize from Human Feedback"
- Bai et al. (2022). "Training a Helpful and Harmless Assistant with RLHF"

## 總結

本實作代表了此資料庫學習進程的巔峰：

**旅程：**
1. **Q-Learning**: 學習價值表
2. **DQN**: 用神經網路擴展價值
3. **PPO**: 用 Actor-Critic 直接學習策略
4. **GRPO-MNIST**: 移除 Critic，使用群組基線
5. **GRPO-MaxText**: 應用於生產規模 LLM

**關鍵成就：**
- ✅ 用 RL 訓練 8B 參數模型
- ✅ 記憶體高效訓練 (無 Critic)
- ✅ 生產基礎設施 (MaxText + vLLM + TPU)
- ✅ 真實世界任務 (數學推理)

**核心教訓：** GRPO 展示了透過巧妙的演算法設計，我們可以透過以推理成本換取訓練記憶體來使大規模模型的 RL 訓練變得可行 - 這在推理引擎高度優化的 LLM 時代是值得的取捨。
