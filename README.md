
# Contextual Bandits in Dynamic Environments

Implements SW-UCB, D-UCB, and Thompson Sampling on a custom dynamic dataset to evaluate adaptability in non-stationary environments. Built for academic research and reproducibility.


## Authors

- [@SumitTeli](https://www.github.com/SumitTeli)
- [@JugalPatel](https://www.github.com/Jugalpatel3981)
## Documentation

[Documentation](https://sumitteli.me/Pattern%20Project/Contextual%20Bandit%20With%20Dynamic%20Environment%20Report.pdf)
## Run Locally

Clone the project

```bash
  git clone https://github.com/SumitTeli/Contextual-Bandits-in-Dynamic-Environments
```

Go to the project directory

```bash
  cd contextual-bandits-project
```

Activate virtual environment

```bash
  .\venv\Scripts\activate

```

(If venv not created)

```bash
 python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Generate the dataset
```bash
python Dataset-dynamic\dataset.py
```

Run the algorithms
```bash
python methods\SW-UCB.py
python methods\D-UCB.py
python methods\ts.py
```

Run the bias check
```bash
python "Biasness check\bias_check_summary.py"
```

Generate result plots
```bash
python ComparisonPlot.py
```
