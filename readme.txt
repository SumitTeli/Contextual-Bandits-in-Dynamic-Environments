-# Activate virtual environment (if it exists)
.\venv\Scripts\activate

# OR (if venv not created yet)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Step 1: Generate the dataset
python Dataset-dynamic\dataset.py

# Step 2: Run all algorithms
python methods\SW-UCB.py
python methods\D-UCB.py
python methods\ts.py

# Step 3: Run bias check
python "Biasness check\bias_check_summary.py"

# Step 4: Generate result plots
python ComparisonPlot.py


# CSV outputs
sw_ucb_results.csv
d_ucb_results.csv
Ts_results.csv
bias_check_summary.csv

# Plots
cumulative_reward_comparison.png
cumulative_regret_comparison.png
accuracy_comparison.png
action_usage_comparison.png
bias_check_context_pca_plot.png
