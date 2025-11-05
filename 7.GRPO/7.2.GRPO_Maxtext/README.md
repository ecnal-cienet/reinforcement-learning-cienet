# Group Relative Policy Optimization (GRPO) 實作 - 大型語言模型訓練 (Qwen3-8B)

## 概述

這是一個將 **Group Relative Policy Optimization (GRPO)** 演算法應用於訓練 **Qwen3-8B 大型語言模型**的實作，目標是在 **GSM8K** (小學數學題) 基準測試上提升模型的數學推理能力。

**與 7.1.GRPO_MNIST 的關鍵差異：**
- **模型規模**：80 億參數 Transformer vs. 小型 MLP
- **任務複雜度**：多步驟數學應用題 vs. 單步數字分類
- **基礎設施**：分散式訓練 (vLLM、MaxText、Tunix) on TPU/GPU vs. 單機 JAX
- **架構**：完整 LLM (注意力機制) vs. 簡單前饋網路
- **生成方式**：自回歸文字生成 vs. 單步分類

## GRPO 是什麼？

GRPO 是一種記憶體高效的強化學習演算法，專為提升 LLM 推理能力設計：

- **無價值網路 (No Value Network)**：與 PPO 不同，GRPO 完全移除 Critic/Value 模型
- **群組相對優勢 (Group-Based Advantages)**：為每個 prompt 生成多個回答，計算相對優勢
- **記憶體高效**：單一模型 (vs. PPO 的 Actor + Critic) 讓訓練更大模型成為可能
- **推理導向**：專為多步驟推理任務設計

### GRPO vs PPO 完整對比

| 特性 | PPO (模組三) | GRPO (本模組) |
|------|-------------|--------------|
| 所需模型 | Actor + Critic | Policy + Reference (凍結) |
| 優勢估計 | GAE (時序差分) | 群組相對 (比較) |
| 記憶體使用 | 高 (2 個網路) | 較低 (1.5 個網路) |
| 最適用於 | 控制任務 | 推理任務 |
| 價值函數 | 學習的基線 | 群組平均作為基線 |

## 技術堆疊

- **Tunix**：GRPO 訓練編排框架
- **vLLM**：高效能推理引擎，用於回應生成
- **MaxText**：基於 JAX 的 LLM 實作，支援 Qwen3、Llama3、Gemma
- **JAX/Flax NNX**：分散式訓練後端
- **Orbax**：檢查點系統
- **Optax**：AdamW 優化器與學習率調度

## 資料集：GSM8K

**GSM8K** (Grade School Math 8K) 包含 8,500 個小學數學應用題，需要多步驟推理：

**範例問題：**
```
問題：「Natalia 在四月賣了 48 個夾子給她的朋友，然後在五月賣出的數量是四月的一半。
      Natalia 在四月和五月總共賣了多少個夾子？」

答案：72
```

**為什麼選擇 GSM8K？**
- 測試多步驟推理與算術能力
- 有明確的正確答案可用於獎勵計算
- 對 LLM 有挑戰性 (需要規劃與計算)

## 訓練流程

### 1. 結構化輸出格式

模型被提示生成特定格式的回應：

```
<reasoning>
步驟 1：Natalia 在四月賣了 48 個夾子
步驟 2：在五月，她賣了 48/2 = 24 個夾子
步驟 3：總計 = 48 + 24 = 72
</reasoning>
<answer>72</answer>
```

這個結構帶來的好處：
- **可解釋性**：可以評估推理過程
- **獎勵工程**：可分別獎勵格式正確性與答案正確性
- **錯誤分析**：識別推理在哪裡出錯

### 2. 獎勵函數

四個互補的獎勵函數 (reinforcement_learning_grpo.py:642-833)：

| 獎勵函數 | 目的 | 獎勵/懲罰 |
|---------|------|---------|
| `match_format_exactly` | 完全符合格式 | +3.0 |
| `match_format_approximately` | 部分格式 (標記計數) | 每個標記 +0.5，錯誤 -0.5 |
| `check_answer` | 答案正確性 (含容錯) | 完全正確: +3.0，接近: +0.5/+0.25，錯誤: -1.0 |
| `check_numbers` | 後備數字提取 | 完全匹配: +1.5 |

**設計理念：**
- 多個訊號引導模型朝向期望的行為
- 格式獎勵鼓勵先建立結構再追求正確性
- 部分分數獎勵 (±10%, ±20%) 鼓勵探索
- 懲罰防止退化行為

### 3. GRPO 訓練迴圈

**高階演算法：**

```
對於每批 prompts：
  1. 為每個 prompt 生成 G 個回應 (NUM_GENERATIONS=2)
  2. 用獎勵函數評估回應
  3. 計算群組優勢：A_i = R_i - mean(R_group)
  4. 計算帶 KL 懲罰的 GRPO loss：
     L = -advantages * log_probs + β * KL(policy || reference)
  5. 限制策略比率 (PPO 風格)：ratio ∈ [1-ε, 1+ε]
  6. 用梯度下降更新策略 (μ 次迭代)
```

**關鍵參數 (reinforcement_learning_grpo.py:184-202)：**
- `NUM_GENERATIONS = 2`：群組大小 (權衡：更好估計 vs. 計算量)
- `BETA = 0.08`：KL 懲罰係數 (控制策略漂移)
- `EPSILON = 0.2`：PPO clip 參數 (訓練穩定性)
- `NUM_ITERATIONS = 1`：每批的優化步驟數

### 4. 模型架構

**策略模型 (Actor)：**
- 基礎：Qwen3-8B transformer (可訓練)
- 精度：bfloat16 以提升記憶體效率
- 訓練：全參數微調 (所有 80 億參數都更新)
- 替代方案：LoRA 適配 (未來增強)

**參考模型 (Reference)：**
- 原始 Qwen3-8B 的凍結副本
- 僅用於 KL 散度計算
- 防止災難性遺忘
- 確保訓練穩定性

**記憶體優化：**
- MaxText remat 策略：「custom」with offloading
- 注意力：dot_product 實作
- vLLM HBM 利用率：20%

## 檔案結構

```
7.2.GRPO_Maxtext/
├── README.md                        # 本檔案
└── reinforcement_learning_grpo.py   # 主訓練腳本
```

## 安裝與設置

### 硬體需求

- **推薦**：單主機 TPU VM (v6e-8 或 v5p-8)
- **替代**：具備足夠記憶體的多 GPU 設置

### 重要提示

⚠️ **此 GRPO on MaxText 實作無法單獨執行**，需要配合特定的 MaxText 版本使用。

### 安裝步驟

```bash
# 1. 複製 MaxText 儲存庫 (AI-Hypercomputer 版本)
cd ~
git clone https://github.com/AI-Hypercomputer/maxtext.git

# 2. 切換到 GRPO 支援的 Draft PR 分支
cd maxtext
# 參考 PR: https://github.com/AI-Hypercomputer/maxtext/pull/2603
# 請依照該 PR 的指示切換到正確的分支

# 3. 執行 MaxText 設置
bash tools/setup/setup.sh

# 4. 啟動虛擬環境
venv_name="maxtext_venv"
source ~/$venv_name/bin/activate

# 5. 安裝 Tunix 和 vLLM 依賴
bash ~/maxtext/src/MaxText/examples/install_tunix_vllm_requirement.sh

# 注意：此安裝可能需要數分鐘
```

**參考資源：**
- [MaxText GRPO PR #2603](https://github.com/AI-Hypercomputer/maxtext/pull/2603) - GRPO 支援的 Draft PR
- [MaxText GRPO 教學](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html)

## 執行程式碼

### 基本執行

```bash
# 確保 MaxText venv 已啟動
source ~/maxtext_venv/bin/activate

# 執行 GRPO 訓練
python reinforcement_learning_grpo.py
```

### 訓練配置

**關鍵超參數 (在腳本中修改)：**

```python
# 資料
TRAIN_FRACTION = 1.0              # 使用 100% 資料 (無驗證集分割)
NUM_BATCHES = 200                 # 要訓練的批次數

# 模型
MODEL_CHECKPOINT_PATH = "gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items"

# GRPO 演算法
NUM_GENERATIONS = 2               # 每個 prompt 的回應數
BETA = 0.08                       # KL 懲罰係數
EPSILON = 0.2                     # PPO clip 參數

# 優化器
LEARNING_RATE = 3e-6              # 峰值學習率
WARMUP_STEPS = MAX_STEPS 的 10%   # LR 暖身期
MAX_GRAD_NORM = 0.1               # 梯度裁剪

# 生成 (訓練期間)
TEMPERATURE = 0.9                 # 高溫度以促進探索
TOP_K = 50                        # Top-k 採樣
TOTAL_GENERATION_STEPS = 1024     # 要生成的最大 token 數
```

### 監控訓練

**TensorBoard：**
```bash
# 在另一個終端機
tensorboard --logdir ~/content/tensorboard/grpo/logs_qwen3 --port=8086
```

**要觀察的指標：**
- `reward/total`：每批次的平均總獎勵
- `reward/[function_name]`：個別獎勵函數的貢獻
- `loss/policy_loss`：GRPO 策略損失
- `loss/kl_divergence`：KL(policy || reference)
- `eval/accuracy`：測試集上的答案準確率
- `eval/format_accuracy`：格式符合率

**JAX 效能分析：**
- 追蹤儲存於：`~/content/jax_traces/grpo/profiles_qwen3/`
- 用以下工具分析：[Perfetto UI](https://ui.perfetto.dev/)

### 檢查點

**自動儲存：**
- 位置：`~/content/ckpts_qwen3/`
- 頻率：每 500 步 (可透過 `SAVE_INTERVAL_STEPS` 配置)
- 最多保留：4 個檢查點 (可透過 `MAX_TO_KEEP` 配置)

## 評估

### 訓練前 vs 訓練後

腳本會自動在訓練前後執行評估：

```python
# 評估指標：
# 1. 答案準確率：完全匹配的百分比
# 2. 部分準確率：在正確答案 ±10% 範圍內的百分比
# 3. 格式準確率：具有正確 <reasoning> 和 <answer> 標籤的百分比
```

### 生成策略

三種評估模式 (reinforcement_learning_grpo.py:278-287)：

```python
GENERATION_CONFIGS = {
    "greedy": {                    # 確定性 (用於指標)
        "temperature": 1e-4,
        "top_k": 1,
        "top_p": 1.0
    },
    "standard": {                  # 平衡採樣
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95
    },
    "liberal": {                   # 高多樣性
        "temperature": 0.85,
        "top_k": 2000,
        "top_p": 1.0
    }
}
```

**使用方式：**
```python
evaluate(test_dataset, rl_cluster, **GENERATION_CONFIGS["greedy"])
```

## 預期結果

### 訓練進度

| 階段 | 答案準確率 | 格式準確率 | 備註 |
|------|----------|-----------|------|
| 訓練前 | ~5-10% | ~20-30% | 基礎 Qwen3-8B 效能 |
| 200 批次後 | ~15-25% | ~60-80% | 先學會格式 |
| 3738 批次後 | ~30-40% | ~90%+ | 完全收斂 (非示範版) |

**注意**：示範使用 `NUM_BATCHES=200` 以快速迭代。生產環境結果請增加至 3738+ 批次。

### 輸出演化範例

**訓練前：**
```
問題：「James 每週兩次寫一封 3 頁的信給 2 個不同的朋友...」

<reasoning>讓我想想...</reasoning>
<answer>我不確定，可能是 10 頁？</answer>
```
**獎勵：** 格式 OK (+3.0)，答案錯誤 (-1.0) = **+2.0**

**訓練後：**
```
問題：「James 每週兩次寫一封 3 頁的信給 2 個不同的朋友...」

<reasoning>
- 每封信 3 頁
- 2 個朋友，所以每次 3 × 2 = 6 頁
- 每週兩次，所以 6 × 2 = 12 頁/週
</reasoning>
<answer>12</answer>
```
**獎勵：** 格式完全正確 (+3.0)，答案完全正確 (+3.0) = **+6.0**

## 程式碼架構

### 主要組件

**步驟 0-1：設置 (第 60-304 行)**
- 匯入函式庫 (Tunix、MaxText、JAX/Flax)
- 配置超參數
- 環境設置 (`SKIP_JAX_PRECOMPILE=1` 用於 vLLM)

**步驟 2-3：工具與資料 (第 306-463 行)**
- `show_hbm_usage()`：監控 TPU/GPU 記憶體
- Tokenizer 設置 (Qwen3)
- 資料集載入與預處理 (GSM8K)

**步驟 4-5：模型 (第 465-638 行)**
- 載入參考模型 (凍結)
- 載入策略模型 (可訓練)
- 兩個模型最初使用相同檢查點
- 每階段的記憶體分析

**步驟 6：獎勵函數 (第 640-833 行)**
- `match_format_exactly()`
- `match_format_approximately()`
- `check_answer()`
- `check_numbers()`

**步驟 7：評估 (第 835-1050 行)**
- `generate_responses()`：vLLM 生成
- `score_responses()`：套用獎勵函數
- `evaluate()`：完整評估流程

**步驟 8：訓練流程 (第 1052-1282 行)**
- `main()`：編排整個工作流程
- 設置優化器 (AdamW + warmup-cosine)
- 建立 RL cluster (Tunix)
- 初始化 GRPO learner
- 執行訓練迴圈
- 訓練前/後評估

### 關鍵設計模式

**MaxText 模型包裝器：**
```python
def get_ref_maxtext_model(config):
    """為 GRPO 訓練建立 TunixMaxTextAdapter"""
    model, mesh = model_creation_utils.create_nnx_model(config)
    with mesh:
        tunix_model = TunixMaxTextAdapter(base_model=model)
    return tunix_model, mesh
```

**RL Cluster 配置：**
```python
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        Role.ACTOR: mesh,          # 策略模型
        Role.REFERENCE: mesh,      # 凍結參考
        Role.ROLLOUT: mesh,        # vLLM 引擎
    },
    rollout_engine="vllm",
    training_config=RLTrainingConfig(...),
    rollout_config=RolloutConfig(...),
)
```

## 進階主題

### LoRA 微調

為提升記憶體效率，考慮使用低秩適配 (Low-Rank Adaptation) 而非全參數微調：

```python
# TODO: 實作 LoRA (見 reinforcement_learning_grpo.py:589)
# 好處：
# - 凍結基礎模型權重
# - 只訓練小型適配矩陣
# - 10-100 倍記憶體減少
# - 更快的訓練
```

### 量化感知訓練

使用 Qwix 進行更低精度：
- 目前：bfloat16
- 替代：int8/int4 配合量化感知訓練
- 權衡：記憶體/速度 vs. 準確性

### 多主機分散式訓練

擴展至多個 TPU/GPU 主機：
- 修改 `skip_jax_distributed_system="false"`
- 為資料/模型平行配置 mesh 形狀
- 與 JAX 分散式初始化協調

### 自訂獎勵函數

設計領域特定獎勵：

```python
def custom_reward(prompts, completions, **context):
    """自訂獎勵函數模板"""
    scores = []
    for completion in completions:
        score = 0.0
        # 你的獎勵邏輯
        scores.append(score)
    return scores

# 加入訓練器：
grpo_trainer = GrpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        check_answer,
        custom_reward,  # 你的函數
    ],
    grpo_config=grpo_config,
)
```

## 疑難排解

### 記憶體問題

**症狀：** OOM 錯誤、HBM 耗盡

**解決方案：**
1. 減少 `NUM_GENERATIONS` (2 → 1)
2. 降低 `BATCH_SIZE` (1 → 1 已是最小值)
3. 減少 `TOTAL_GENERATION_STEPS` (1024 → 512)
4. 增加 `rollout_vllm_hbm_utilization` (0.2 → 0.3)
5. 啟用 LoRA 微調

### vLLM 初始化失敗

**症狀：** `AsyncioEventLoop already running`

**解決方案：**
```python
import nest_asyncio
nest_asyncio.apply()
```

### 訓練準確率低

**症狀：** 訓練後無改善

**解決方案：**
1. 增加 `NUM_BATCHES` (200 → 3738)
2. 增加 `NUM_GENERATIONS` (2 → 4)
3. 調整獎勵函數權重
4. 降低 `LEARNING_RATE` (3e-6 → 1e-6)
5. 在 TensorBoard 檢查獎勵訊號

### JAX 分散式錯誤

**症狀：** `skip_jax_distributed_system` 警告

**解決方案：**
- 目前配置針對單主機優化
- 多主機請參考 MaxText 分散式訓練文件

## 學習成果

完成本模組後，你應該理解：

1. **GRPO 演算法**：群組相對優勢如何取代學習的價值函數
2. **LLM RL 訓練**：訓練大型模型的基礎設施 (vLLM、MaxText、Tunix)
3. **獎勵工程**：為複雜任務設計多面向獎勵
4. **分散式訓練**：JAX mesh 配置、模型/資料平行
5. **結構化生成**：使用格式約束引導 LLM 輸出
6. **評估**：用量化指標衡量推理改善

## 比較：GRPO_MNIST vs GRPO_Maxtext

| 面向 | 7.1.GRPO_MNIST | 7.2.GRPO_Maxtext |
|------|----------------|------------------|
| **任務** | 數字分類 | 數學應用題 |
| **模型** | 3 層 MLP (~1 萬參數) | Qwen3-8B transformer (80 億參數) |
| **輸入** | 28×28 圖片 → 784D 向量 | 文字 prompt → tokens |
| **輸出** | 10D logits (數字 0-9) | 自回歸文字 |
| **動作空間** | 離散 (10 類) | 離散 (詞彙量 ~15 萬) |
| **訓練** | 單裝置，<1 分鐘 | TPU/GPU 叢集，數小時 |
| **獎勵** | 正確類別 (+1/-1) | 多函數 (格式+答案) |
| **基礎設施** | 純 JAX/Flax | Tunix + vLLM + MaxText |
| **複雜度** | 教育性玩具範例 | 生產級 LLM 訓練 |

## 參考資料

- [MaxText GRPO 教學](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html)
- [GRPO 論文](https://arxiv.org/abs/2402.03300) - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
- [GSM8K 資料集](https://arxiv.org/abs/2110.14168) - "Training Verifiers to Solve Math Word Problems"
- [vLLM 文件](https://docs.vllm.ai/)
- [Tunix 文件](https://github.com/google/maxtext/tree/main/src/tunix)
- [Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-8B)

## 下一步

1. **實驗超參數**：增加 `NUM_BATCHES`，調整 `BETA`
2. **嘗試不同模型**：Llama3-8B、Gemma-7B (修改 `model_name` 配置)
3. **自訂資料集**：將 GRPO 應用於編碼任務、推理基準測試
4. **實作 LoRA**：記憶體高效適配
5. **多步驟推理**：分析失敗案例，改善獎勵函數
6. **部署**：匯出訓練好的模型用於推理

---

**作者註：** 這是一個進階模組，展示 LLM 強化學習的最新技術。首次學習者建議從 7.1.GRPO_MNIST 開始，掌握核心 GRPO 概念後再處理這個生產級實作。
