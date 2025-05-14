import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

dataset_path = "Dataset/dataset.csv"
df = pd.read_csv(dataset_path)

bias_report = {}

phase_dist = df.groupby("phase")["optimal_action"].value_counts(normalize=True).unstack().fillna(0)
for phase in phase_dist.index:
    for arm in phase_dist.columns:
        key = f"{phase}_optimal_arm_{arm}_ratio"
        bias_report[key] = phase_dist.loc[phase, arm]

global_dist = df["optimal_action"].value_counts(normalize=True)
for arm in global_dist.index:
    bias_report[f"global_optimal_arm_{arm}_ratio"] = global_dist.loc[arm]

avg_probs = df[["true_prob_music", "true_prob_sports", "true_prob_tech"]].mean()
for name, val in avg_probs.items():
    bias_report[f"avg_{name}"] = val

df["tech_dominant"] = df["context_tech"] > 0.5
tech_bias_dist = df[df["tech_dominant"]]["optimal_action"].value_counts(normalize=True)
for arm in tech_bias_dist.index:
    bias_report[f"tech_dominant_optimal_arm_{arm}_ratio"] = tech_bias_dist.loc[arm]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[["context_music", "context_sports", "context_tech"]])
df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]
bias_report["pca_1_variance_ratio"] = pca.explained_variance_ratio_[0]
bias_report["pca_2_variance_ratio"] = pca.explained_variance_ratio_[1]

output_folder = "Biasness check"
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, "bias_check_summary.csv")
pd.DataFrame([bias_report]).to_csv(csv_path, index=False)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["pca_1"], df["pca_2"], c=df["optimal_action"], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Optimal Action")
plt.title("Context Distribution by Optimal Action (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plot_path = os.path.join(output_folder, "bias_check_context_pca_plot.png")
plt.savefig(plot_path)
plt.close()

print(f"Bias report saved to '{csv_path}'")
print(f"PCA plot saved to '{plot_path}'")
