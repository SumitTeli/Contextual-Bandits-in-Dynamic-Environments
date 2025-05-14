import pandas as pd
import numpy as np
import math
import os

DISCOUNT_FACTOR = 0.95        
CONFIDENCE_SCALE = 0.5        
REWARD_BOUND = 1.0            
NUM_ARMS = 3


df = pd.read_csv("Dataset/dataset.csv")


history = {i: [] for i in range(NUM_ARMS)}  
cumulative_rewards = []
cumulative_regret = []
total_reward = 0
correct_choices = 0


chosen_arms = []
optimal_arms = []
rewards = []


for index, row in df.iterrows():
    t = index + 1

    true_probs = {
        0: row["true_prob_music"],
        1: row["true_prob_sports"],
        2: row["true_prob_tech"]
    }
    optimal_arm = int(row["optimal_action"])

    
    arm_values = []
    for arm in range(NUM_ARMS):
        discounted_sum = 0.0
        discounted_count = 0.0

        for round_idx, reward in history[arm]:
            weight = DISCOUNT_FACTOR ** (t - round_idx)
            discounted_sum += weight * reward
            discounted_count += weight

        mean = discounted_sum / discounted_count if discounted_count > 0 else 0.0
        arm_values.append((mean, discounted_count))

    total_count = sum(dc for _, dc in arm_values)
    scores = []

    for arm in range(NUM_ARMS):
        mean, count = arm_values[arm]
        if count > 0:
            bonus = REWARD_BOUND * math.sqrt((CONFIDENCE_SCALE * math.log(total_count + 1)) / count)
        else:
            bonus = float("inf")
        scores.append(mean + bonus)

    selected_arm = int(np.argmax(scores))
    selected_prob = true_probs[selected_arm]
    reward = np.random.binomial(1, selected_prob)

    history[selected_arm].append((t, reward))

    total_reward += reward
    regret = true_probs[optimal_arm] - selected_prob
    cumulative_regret.append((cumulative_regret[-1] if cumulative_regret else 0) + regret)
    cumulative_rewards.append(total_reward)
    correct_choices += int(selected_arm == optimal_arm)

    chosen_arms.append(selected_arm)
    optimal_arms.append(optimal_arm)
    rewards.append(reward)

print("D-UCB Simulation Complete")
print(f"Total Reward: {total_reward}")
print(f"Accuracy: {correct_choices / len(df):.3f}")
print(f"Final Cumulative Regret: {cumulative_regret[-1]:.3f}")

output_df = pd.DataFrame({
    "round": range(1, len(df) + 1),
    "action": chosen_arms,
    "optimal_action": optimal_arms,
    "reward": rewards,
    "cumulative_reward": cumulative_rewards,
    "cumulative_regret": cumulative_regret
})
output_path = "d_ucb_results_.csv"
output_df.to_csv(output_path, index=False)
print(f"Results saved to '{output_path}'")
