#!/usr/bin/env python

# Q_Learning.py
# 實作一個簡單的 Q-Learning 範例
# 環境：4x4 Grid World
# 狀態空間 (state space): 16 個離散狀態 (0 到 15)
# 動作空間 (action space): 4 個動作 (上、下、左、右)
# 獎勵函數 (reward function):
#   - 到達目標狀態 (狀態 15) 獎勵 +100
#   - 每移動一步扣 -1

import numpy as np


def main():
    # Env Setup
    GRID_SIZE = 4
    NUM_STATES = GRID_SIZE * GRID_SIZE  # 16 個狀態 (0 到 15)
    NUM_ACTIONS = 4  # 4 actions: 0: up, 1: down, 2: left, 3: right
    GOAL_STATE = NUM_STATES - 1  # 目標在 (3, 3)，也就是狀態 15
    START_STATE = 0  # 起點在 (0, 0)，也就是狀態 0

    # Q-Learning Hyperparameters
    ALPHA = 0.1  # 學習率 (α)：我們對新資訊的信任程度
    GAMMA = 0.99  # 折扣因子 (γ)：我們對「未來獎勵」的重視程度
    NUM_EPISODES = 20000  # 總共要玩的遊戲次數 (訓練回合)
    MAX_STEPS_PER_EPISODE = 100  # 每回合最多走 100 步

    # Epsilon-Greedy Parameters (ε) 用於「探索 vs. 利用」
    EPSILON = 1.0  # 初始 ε 值 (完全探索)
    MAX_EPSILON = 1.0  # 最大 ε 值
    MIN_EPSILON = 0.01  # 最終 Epsilon (1% 探索)
    DECAY_EPISODES = 10000  # Epsilon 衰退到 0.01 所需的回合數

    # 初始化 Q 表格，形狀為 (16, 4)，初始值全為 0
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    print("\n--- 學習前 Q-Table (片段) ---")
    print(q_table)

    def step(state, action) -> tuple:
        """
        執行一個動作，並回傳 (新狀態, 獎勵, 是否完成)
        :param state: 目前狀態
        :param action: 要執行的動作
        :return: (new_state, reward, done)
        """
        row = state // GRID_SIZE  # 取整數：計算目前所在的行
        col = state % GRID_SIZE  # 取餘數：計算目前所在的列

        # 根據動作更新位置並且確保不會超出邊界
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(GRID_SIZE - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(GRID_SIZE - 1, col + 1)

        new_state = row * GRID_SIZE + col

        # 檢查是否到達目標狀態
        if new_state == GOAL_STATE:
            reward = 100
            done = True
        else:
            reward = -1  # 每走一步都扣 1 分
            done = False

        return new_state, reward, done

    # Q-Learning Training Loop
    for episode in range(NUM_EPISODES):
        state = START_STATE
        done = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            # Epsilon-Greedy Action Selection
            if np.random.uniform(0, 1) < EPSILON:
                action = np.random.choice(NUM_ACTIONS)  # 探索: 隨機選擇動作
            else:
                action = np.argmax(q_table[state, :])  # 利用: 選擇 Q 值最高的動作

            # 執行動作並從環境中獲取回饋
            new_state, reward, done = step(state, action)

            # ##################################################
            # ### 這是 Q-Learning 最核心的更新公式 ###
            # ##################################################

            # 1. 找出S' (new_state) 的未來最佳價值: max Q(S', a')
            feature_best_q = np.max(q_table[new_state, :])

            # 2. 計算 TD 目標值 (我們希望Q(S, a) 接近的值)
            # TD Target = R + γ * max Q(S', a')
            target = reward + (GAMMA * feature_best_q)

            # 3. 更新 Q-table
            # Q(S,A) = Q(S,A) + α * (TD_Target - Q(S,A))
            q_table[state, action] = q_table[state, action] + ALPHA * (
                target - q_table[state, action]
            )

            # 更新狀態
            state = new_state

            if done:
                break  # 如果到達目標，提前結束這回合

        # Epsilon 衰退
        # 隨著訓練進行，我們讓 Agent 越來越少「探索」，更多「利用」
        if episode < DECAY_EPISODES:
            EPSILON = MAX_EPSILON - (MAX_EPSILON - MIN_EPSILON) * (
                episode / DECAY_EPISODES
            )
        else:
            EPSILON = MIN_EPSILON

        # 每 2000 回合輸出一次目前的 Epsilon 值
        if (episode + 1) % 2000 == 0:
            print(f"Episode: {episode + 1}, Epsilon: {EPSILON:.4f}")

    print("Training completed!")

    # 顯示結果（學到的策略）
    print("\n--- 最終 Q-Table (片段) ---")
    print(q_table)

    print("\n--- 學習到的最佳策略 (Policy) ---")
    # 0:上(↑), 1:下(↓), 2:左(←), 3:右(→)
    actions_map = ["↑", "↓", "←", "→"]
    policy_grid = ""
    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            state = r * GRID_SIZE + c
            if state == GOAL_STATE:
                row_str += " G "
                continue
            # 從 Q-Table 中找出這個狀態下 Q 值最高的動作
            best_action_idx = np.argmax(q_table[state, :])
            row_str += f" {actions_map[best_action_idx]} "
        policy_grid += row_str + "\n"

    print(policy_grid)


if __name__ == "__main__":
    main()
