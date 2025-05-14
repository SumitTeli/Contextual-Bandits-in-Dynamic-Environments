import pandas as pd
import matplotlib.pyplot as plt


df_sw = pd.read_csv("sw_ucb_results.csv")
df_du = pd.read_csv("d_ucb_results.csv")
df_ts = pd.read_csv("Ts_results.csv")

# ---------- Plot 1: Cumulative Reward ----------
plt.figure(figsize=(10, 6))
plt.plot(df_sw["round"], df_sw["cumulative_reward"], label="SW-UCB")
plt.plot(df_du["round"], df_du["cumulative_reward"], label="D-UCB")
plt.plot(df_ts["round"], df_ts["cumulative_reward"], label="Thompson Sampling")
plt.title("Cumulative Reward Over Time")
plt.xlabel("Round")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_reward_comparison.png")
plt.close()

# ---------- Plot 2: Cumulative Regret ----------
plt.figure(figsize=(10, 6))
plt.plot(df_sw["round"], df_sw["cumulative_regret"], label="SW-UCB")
plt.plot(df_du["round"], df_du["cumulative_regret"], label="D-UCB")
plt.plot(df_ts["round"], df_ts["cumulative_regret"], label="Thompson Sampling")
plt.title("Cumulative Regret Over Time")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_regret_comparison.png")
plt.close()

# ---------- Plot 3: Rolling Accuracy (Window = 50) ----------
window = 50
df_sw["accuracy"] = (df_sw["action"] == df_sw["optimal_action"]).rolling(window).mean()
df_du["accuracy"] = (df_du["action"] == df_du["optimal_action"]).rolling(window).mean()
df_ts["accuracy"] = (df_ts["action"] == df_ts["optimal_action"]).rolling(window).mean()

plt.figure(figsize=(10, 6))
plt.plot(df_sw["round"], df_sw["accuracy"], label="SW-UCB")
plt.plot(df_du["round"], df_du["accuracy"], label="D-UCB")
plt.plot(df_ts["round"], df_ts["accuracy"], label="Thompson Sampling")
plt.title(f"Rolling Accuracy (Window={window})")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.close()

# ---------- Plot 4: Action Usage ----------
action_sw = df_sw["action"].value_counts(normalize=True).sort_index()
action_du = df_du["action"].value_counts(normalize=True).sort_index()
action_ts = df_ts["action"].value_counts(normalize=True).sort_index()

action_usage = pd.DataFrame({
    "SW-UCB": action_sw,
    "D-UCB": action_du,
    "Thompson Sampling": action_ts
}).fillna(0)

action_usage.index = [f"Arm {i}" for i in action_usage.index]

action_usage.plot(kind="bar", figsize=(10, 6))
plt.title("Action Usage Distribution")
plt.ylabel("Proportion of Plays")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("action_usage_comparison.png")
plt.close()

# ---------- Plot 5: Instantaneous Regret ----------
df_sw["instant_regret"] = df_sw["cumulative_regret"].diff().fillna(df_sw["cumulative_regret"])
df_du["instant_regret"] = df_du["cumulative_regret"].diff().fillna(df_du["cumulative_regret"])
df_ts["instant_regret"] = df_ts["cumulative_regret"].diff().fillna(df_ts["cumulative_regret"])

plt.figure(figsize=(12, 6))
plt.plot(df_sw["round"], df_sw["instant_regret"], label="SW-UCB", alpha=0.7)
plt.plot(df_du["round"], df_du["instant_regret"], label="D-UCB", alpha=0.7)
plt.plot(df_ts["round"], df_ts["instant_regret"], label="Thompson Sampling", alpha=0.7)
plt.title("Per-Round (Instantaneous) Regret")
plt.xlabel("Round")
plt.ylabel("Instantaneous Regret")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("instantaneous_regret_comparison.png")
plt.close()
