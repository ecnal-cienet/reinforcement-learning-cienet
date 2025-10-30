#!/usr/bin/env python

import jax
import jax.numpy as jnp
from flax import nnx
import optax

# 匯入 TFDS (真實資料集)
import tensorflow_datasets as tfds
import tensorflow as tf
# 匯入 tfd.Categorical (離散動作分佈)
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

# --- 超參數 ---
LEARNING_RATE = 1e-4  # Actor 的學習率
NUM_EPOCHS = 10  # 總共要訓練幾輪
BATCH_SIZE = 1024  # **這就是 GRPO 的「群組大小 (G)」**
CLIP_EPSILON = 0.2  # PPO 的「安全鎖」 (ε)


def get_dataset(batch_size: int) -> tfds.as_numpy:
    """
    載入並預處理 MNIST 資料集
    參數:
        batch_size: 每一批次的大小
    回傳:
        一個可以產生批次資料的 Python 迭代器
    """
    # 1. 載入 MNIST
    ds = tfds.load("mnist", split="train", as_supervised=True)

    # 2. 定義預處理 (Preprocessing) 函式
    def preprocess(image, label):
        # (a) 將 (28, 28, 1) 的圖片「扁平化」成 (784,) 的向量
        image = tf.reshape(image, (-1,))
        # (b) 將 0-255 的整數，正規化 (Normalize) 到 0.0-1.0 的浮點數
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # 3. 應用預處理
    ds = ds.map(preprocess)

    # 4. 打亂 (Shuffle) 和 批次 (Batch)
    #    這就是我們的「資料環境」
    ds = ds.shuffle(buffer_size=10_000, seed=42)
    ds = ds.batch(batch_size)
    #    prefetch(1) 可以在 GPU 訓練時，讓 CPU 提前準備下一批資料
    ds = ds.prefetch(1)

    # 將 tfds 物件轉換為一個 Python 迭代器
    return tfds.as_numpy(ds)


# --- 1. 定義「演員 (Actor)」網路 ---
#   它的工作：學習 π(a|s) (策略)
class Actor(nnx.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        """
        一個 MLP，用於 MNIST 分類
        Input(784) -> 128 -> ReLU -> 128 -> ReLU -> Output(10)

        Args:
            in_features (int): 狀態空間維度 (784)
            out_features (int): 動作空間維度 (10)
            rngs (nnx.Rngs): 隨機種子
        """
        self.fc1 = nnx.Linear(in_features, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 128, rngs=rngs)
        # 輸出 10 個「logits」(原始分數)，用於 Categorical 分佈
        self.fc_out = nnx.Linear(128, out_features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tfd.Categorical:
        """
        前向傳播，輸入狀態，輸出動作分佈
        Args:
            x (jnp.ndarray): 狀態輸入，形狀為 (batch_size, 784)
        Returns:
            tfd.Categorical: 動作的機率分佈 (10 個動作 (0-9) 的機率分佈)
        """
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        logits = self.fc_out(x)

        # 2. 回傳一個「機率分佈」
        #    tfd.Categorical 會自動對 logits 做 softmax
        return tfd.Categorical(logits=logits)


# --- 2. 建立 GRPO Agent (無 Critic 版本) ---
class GRPOAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        rng_key: jax.Array,
    ):
        """
        建立 GRPO Agent (只有 Actor)
        Args:
            state_dim (int): 狀態維度 (784)
            action_dim (int): 動作維度 (10)
            rng_key (jax.Array): 隨機種子
        """
        # 1. 建立 Actor 網路
        # --- 1. 建立 Actor (演員) 和它的優化器 ---
        self.actor = Actor(state_dim, action_dim, rngs=nnx.Rngs(rng_key))

        self.actor_optimizer = nnx.Optimizer(
            self.actor, optax.adam(learning_rate=LEARNING_RATE)
        )

        self.rng_stream = nnx.Rngs(jax.random.PRNGKey(42))

    def select_actions_and_log_probs(self, batch_states: jax.Array):
        """
        這是在「收集 (Rollout)」階段呼叫的函式。
        它會回傳「動作」和「Log機率」。

        Args:
            batch_states (jax.Array): 一批 (G=1024) 的狀態

        Returns:
            Tuple[jax.Array, jax.Array]: (採樣的動作, 動作的 Log 機率)
        """
        # 1. 呼叫 Actor 取得「機率分佈」
        action_dist = self.actor(batch_states)

        # 2. 採樣一個動作 (為 G=1024 筆資料中的每一筆採樣)
        rng_key = self.rng_stream.sampler()
        actions = action_dist.sample(seed=rng_key)

        # 3. 計算這些動作的「Log 機率」
        log_probs = action_dist.log_prob(actions)

        # 4. 回傳 JAX arrays (我們將在主迴圈中處理它們)
        return actions, log_probs

    def train_step(
        self,
        batch_states: jax.Array,
        batch_actions: jax.Array,
        batch_log_probs_old: jax.Array,
        batch_advantages: jax.Array,
    ):
        """
        GRPO 的核心學習步驟 (只有 Actor)。

        Args:
            batch_states (jax.Array): G 筆狀態 (圖片)
            batch_actions (jax.Array): G 筆動作 (猜測的數字)
            batch_log_probs_old (jax.Array): G 筆「舊的」Log 機率
            batch_advantages (jax.Array): G 筆「相對優勢」(+0.85 / -0.25)
        """

        # --- 訓練「演員 (Actor)」 (PPO Clipping Loss) ---
        def actor_loss_fn(actor_model: Actor):
            # (1) 取得「新的」機率分佈和「新的」Log 機率
            action_dist_new = actor_model(batch_states)
            # 評估「新策略」對「舊動作」的看法
            log_probs_new = action_dist_new.log_prob(batch_actions)

            # (2) 計算「策略比例 (Policy Ratio)」
            #     Ratio = π_new / π_old
            ratio = jnp.exp(log_probs_new - batch_log_probs_old)

            # (3) 計算「未裁剪的」Loss
            loss_unclipped = batch_advantages * ratio

            # (4) PPO 核心：計算「裁剪後的 (Clipped)」Loss
            ratio_clipped = jnp.clip(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
            loss_clipped = batch_advantages * ratio_clipped

            # (5) 取「兩者中較小」的那個 (PPO 的悲觀原則)
            loss = -jnp.mean(jnp.minimum(loss_unclipped, loss_clipped))

            return loss

        # (6) 計算梯度並更新 Actor
        # 我們使用 nnx.value_and_grad 來獲取 loss 和 grads
        # (在這個專案中我們不監控 loss 值，所以用 _ 忽略它)
        _, actor_grads = nnx.value_and_grad(actor_loss_fn)(self.actor)
        self.actor_optimizer.update(actor_grads)


# --- 3. 主訓練迴圈 ---
def main():
    print("開始 GRPO on MNIST 訓練...")

    # --- A. 初始化 ---
    STATE_DIM = 784  # 28 * 28
    ACTION_DIM = 10  # 0-9

    main_rng = jax.random.PRNGKey(42)
    agent = GRPOAgent(STATE_DIM, ACTION_DIM, rng_key=main_rng)

    # 載入我們的「真實資料環境」
    mnist_dataset = get_dataset(BATCH_SIZE)  # BATCH_SIZE 就是 G

    # --- B. GRPO 訓練大迴圈 ---
    for epoch in range(NUM_EPOCHS):
        total_correct = 0
        total_samples = 0

        # 迭代 (Iterate) 我們的 MNIST 資料集
        # 每一批 (batch) 就是一個「群組 (Group)」
        for batch in mnist_dataset:

            # 1. 解開資料
            #    batch_images 的形狀是 (G, 784)
            #    batch_labels 的形狀是 (G,)
            batch_images, batch_labels = batch

            # --- 階段 1: 收集 (Rollout) ---
            #    (在單步問題中，Rollout 和資料載入是同時的)

            # 取得「猜測的動作」和「舊 Log 機率」
            # 這一步是在 JAX 中完成的
            batch_actions, batch_log_probs_old = agent.select_actions_and_log_probs(
                batch_images
            )

            # --- 階段 2: 計算「獎勵」和「基線」 (GRPO 核心) ---

            # (A) 計算「獎勵 (Rewards)」
            #     我們在 JAX 中計算，以保持高效能
            @jax.jit
            def calculate_rewards(actions, labels):
                # 答對 +1.0，答錯 0.0
                return jnp.where(actions == labels, 1.0, 0.0)

            batch_rewards = calculate_rewards(batch_actions, batch_labels)

            # (B) 計算「群組基線 (Group Baseline)」
            #     這就是 GRPO 的「Critic 替代品」
            baseline = jnp.mean(batch_rewards)

            # (C) 計算「相對優勢 (Relative Advantage)」
            #     Advantage = (個體獎勵) - (群組平均獎勵)
            batch_advantages = batch_rewards - baseline

            # (D) [優化] 標準化 Advantage (和 PPO 一樣)
            adv_mean = jnp.mean(batch_advantages)
            adv_std = jnp.std(batch_advantages) + 1e-8
            batch_advantages = (batch_advantages - adv_mean) / adv_std

            # --- 階段 3: 學習 (Learn) ---
            # 呼叫我們剛剛定義的 PPO 引擎室
            agent.train_step(
                batch_images, batch_actions, batch_log_probs_old, batch_advantages
            )

            # --- 階段 4: 監控 (Monitor) ---
            total_correct += jnp.sum(batch_rewards)  # (因為 reward 是 1.0)
            total_samples += len(batch_labels)

        # 每個 Epoch 結束後，報告準確率
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, " f"Accuracy: {accuracy * 100:.2f}%")

    print("--- 訓練完成！ ---")


# 執行主函式
if __name__ == "__main__":
    main()
