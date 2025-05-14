import numpy as np
import pandas as pd
import os
import random


NUM_ROUNDS = 1500
NUM_ARMS = 3
CONTEXT_DIM = 3
FADE_LENGTH = 30 
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

arm_labels = ['music', 'sports', 'tech']


def generate_random_context():
    context = np.random.rand(CONTEXT_DIM)
    return context / context.sum()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def generate_weight_matrix(dominant_arm_index):
    W = np.random.randn(CONTEXT_DIM, NUM_ARMS) * 0.5
    boost = np.random.uniform(1.8, 2.4)
    W[:, dominant_arm_index] += boost
    return W


def generate_drift_plan(total_rounds, min_stay=150, max_stay=300):
    arm_sequence = []
    current_round = 0
    history = []

    while current_round < total_rounds:
        candidate_arms = list(range(NUM_ARMS))
        if len(history) >= 2:
            for recent_arm in history[-2:]:
                if recent_arm in candidate_arms:
                    candidate_arms.remove(recent_arm)

        if not candidate_arms:
            candidate_arms = list(range(NUM_ARMS))

        next_arm = random.choice(candidate_arms)
        stay_duration = random.randint(min_stay, max_stay)

        arm_sequence.append((next_arm, stay_duration))
        current_round += stay_duration
        history.append(next_arm)

    return arm_sequence


drift_plan = generate_drift_plan(NUM_ROUNDS)
rows = []
round_counter = 1

for i in range(len(drift_plan)):
    arm_idx, duration = drift_plan[i]
    next_arm_idx, _ = drift_plan[i + 1] if i + 1 < len(drift_plan) else (arm_idx, 0)

    W_start = generate_weight_matrix(arm_idx)
    W_end = generate_weight_matrix(next_arm_idx)

    for j in range(duration):
        if j >= duration - FADE_LENGTH and i + 1 < len(drift_plan):
            alpha = (j - (duration - FADE_LENGTH)) / FADE_LENGTH
            W = (1 - alpha) * W_start + alpha * W_end
            phase_label = f"{arm_labels[arm_idx]}_to_{arm_labels[next_arm_idx]}_fade"
        else:
            W = W_start
            phase_label = f"{arm_labels[arm_idx]}"

        context = generate_random_context()
        raw_scores = W.T @ context
        reward_probs = softmax(raw_scores)

        action_taken = np.random.choice(NUM_ARMS)
        reward_received = np.random.binomial(1, reward_probs[action_taken])
        observed_prob = reward_probs[action_taken]
        optimal_action = int(np.argmax(reward_probs))

        row = {
            'round': round_counter,
            'context_music': context[0],
            'context_sports': context[1],
            'context_tech': context[2],
            'action_taken': action_taken,
            'reward': reward_received,
            'true_prob_music': reward_probs[0],
            'true_prob_sports': reward_probs[1],
            'true_prob_tech': reward_probs[2],
            'observed_true_prob': observed_prob,
            'optimal_action': optimal_action,
            'phase': phase_label
        }

        rows.append(row)
        round_counter += 1

        if round_counter > NUM_ROUNDS:
            break
    if round_counter > NUM_ROUNDS:
        break

output_folder = "Dataset"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "contextual_bandit_dynamic_dataset.csv")

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
print(f"dynamic dataset saved to: {output_path}")
