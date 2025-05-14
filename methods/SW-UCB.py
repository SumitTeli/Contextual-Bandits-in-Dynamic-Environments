import pandas as pd
import numpy as np
import math
import os


WINDOW_SIZE = 50            
REWARD_BOUND = 1.0        
CONFIDENCE_SCALE = 0.5    
NUM_ARMS = 3

df = pd.read_csv("Dataset/dataset.csv")


reward_history = {arm: [] for arm in range(NUM_ARMS)}  
cumulative_rewards = []
cumulative_regret = []
correct_decisions = 0
total_reward = 0


chosen_arms = []
optimal_arms = []
rewards = []


for index, row in df.iterrows():
    round_number = index + 1

    
    context = np.array([
        row["context_music"],
        row["context_sports"],
        row["context_tech"]
    ])

   
    true_probs = {
        0: row["true_prob_music"],
        1: row["true_prob_sports"],
        2: row["true_prob_tech"]
    }
    optimal_arm = int(row["optimal_action"])

   
    ucb_scores = []
    for arm in range(NUM_ARMS):
        recent_rewards = [
            reward for past_round, reward in reward_history[arm]
            if round_number - past_round < WINDOW_SIZE
        ]
        count = len(recent_rewards)
        mean_reward = np.mean(recent_rewards) if count > 0 else 0.0

        if count > 0:
            confidence = REWARD_BOUND * math.sqrt(
                (CONFIDENCE_SCALE * math.log(min(round_number, WINDOW_SIZE))) / count
            )
        else:
            confidence = float('inf')

        ucb_scores.append(mean_reward + confidence)

    selected_arm = int(np.argmax(ucb_scores))
    selected_prob = true_probs[selected_arm]
    reward = np.random.binomial(1, selected_prob)

    
    reward_history[selected_arm].append((round_number, reward))
    reward_history[selected_arm] = reward_history[selected_arm][-WINDOW_SIZE:]

    
    total_reward += reward
    regret = true_probs[optimal_arm] - selected_prob
    cumulative_regret.append((cumulative_regret[-1] if cumulative_regret else 0) + regret)
    cumulative_rewards.append(total_reward)
    correct_decisions += int(selected_arm == optimal_arm)

    
    chosen_arms.append(selected_arm)
    optimal_arms.append(optimal_arm)
    rewards.append(reward)


print("SW-UCB Simulation Complete")
print(f"Total Reward: {total_reward}")
print(f"Accuracy: {correct_decisions / len(df):.3f}")
print(f"Final Cumulative Regret: {cumulative_regret[-1]:.3f}")


results_df = pd.DataFrame({
    "round": range(1, len(df) + 1),
    "action": chosen_arms,
    "optimal_action": optimal_arms,
    "reward": rewards,
    "cumulative_reward": cumulative_rewards,
    "cumulative_regret": cumulative_regret
})
results_df.to_csv("sw_ucb_results.csv", index=False)
print("Results saved to 'sw_ucb_results.csv'")
