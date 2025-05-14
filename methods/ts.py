import pandas as pd
import numpy as np
import os


NUM_ARMS = 3
CONTEXT_DIM = 3
LAMBDA = 1.0     
V_SQUARED = 0.1  

df = pd.read_csv("Dataset/dataset.csv")


A = [LAMBDA * np.identity(CONTEXT_DIM) for _ in range(NUM_ARMS)]  
b = [np.zeros((CONTEXT_DIM, 1)) for _ in range(NUM_ARMS)]          

cumulative_rewards = []
cumulative_regret = []
total_reward = 0
correct_choices = 0


chosen_arms = []
optimal_arms = []
rewards = []


for index, row in df.iterrows():
    x = np.array([
        row["context_music"],
        row["context_sports"],
        row["context_tech"]
    ]).reshape(-1, 1)

    true_probs = {
        0: row["true_prob_music"],
        1: row["true_prob_sports"],
        2: row["true_prob_tech"]
    }
    optimal_arm = int(row["optimal_action"])

    sampled_values = []
    for arm in range(NUM_ARMS):
        A_inv = np.linalg.inv(A[arm])
        theta_hat = A_inv @ b[arm]
        theta_sample = np.random.multivariate_normal(theta_hat.ravel(), V_SQUARED * A_inv)
        sampled_reward = float(x.T @ theta_sample.reshape(-1, 1))
        sampled_values.append(sampled_reward)

    selected_arm = int(np.argmax(sampled_values))
    selected_prob = true_probs[selected_arm]
    reward = np.random.binomial(1, selected_prob)

    A[selected_arm] += x @ x.T
    b[selected_arm] += reward * x

    total_reward += reward
    regret = true_probs[optimal_arm] - selected_prob
    cumulative_regret.append((cumulative_regret[-1] if cumulative_regret else 0) + regret)
    cumulative_rewards.append(total_reward)
    correct_choices += int(selected_arm == optimal_arm)

    chosen_arms.append(selected_arm)
    optimal_arms.append(optimal_arm)
    rewards.append(reward)


print("Thompson Sampling Simulation Complete")
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
output_df.to_csv("Ts_results.csv", index=False)
print("Results saved to 'ts_resultsrealworld.csv'")
