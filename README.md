# SemInd: Semantic Inductive Attribute Selection for Zero-Shot Learning

A unified framework combining genetic algorithms (Gazela) and semantic inductive methods (HAIS) for zero-shot learning attribute selection.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Semantic Inductive Attribute Selection for Zero-Shot Learning},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={X},
  pages={X--X}
}
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv env_semind
source env_semind/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Quick Start

### 1. Genetic Algorithm (GA) Experiments

Run genetic algorithm optimization on AWA2 dataset:

```bash
python ga_deap.py --dataset AWA2 --generations 300 --population-size 50 --seed 42
```

### 2. Random Forest Selector (RFS) Baseline

Run RFS baseline comparison:

```bash
python rfs.py --dataset AWA2 --n-estimators 100 --seed 42
```

## Reproducible Experiments

To reproduce the results from our paper, use the exact seeds from our experiments:

### Seeds Used in GA Experiments

| Dataset | Algorithm | Seeds (20 runs) |
|---------|-----------|-----------------|
| AWA2 | eaMuPlusLambda | [1593, 4782, 6139, 6598, 4680, 2856, 6203, 4906, 2024, 3562, 9876, 1604, 3064, 2459, 3841, 1203, 4454, 2122, 1532, 9999] |
| CUB | eaMuPlusLambda | [8326, 5855, 42, 6591, 2122, 5923, 1593, 7727, 9876, 9274, 6238, 2024, 6203, 5678, 6187, 6346, 1234, 3491, 8742, 6139] |
| FLO | eaMuPlusLambda | [6346, 3562, 1121, 5855, 1337, 7815, 3141, 8765, 8742, 4680, 2785, 4906, 2024, 2459, 4454, 7051, 9175, 3915, 1947, 8123] |

Example reproduction command:
```bash
python ga_deap.py --dataset AWA2 --seed 1593 --generations 300 --population-size 50
```

## Key Parameters

- `--dataset`: Dataset to use (AWA2, CUB, FLO, APY, SUN)
- `--generations`: Number of GA generations (default: 300)
- `--population-size`: GA population size (default: 50)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable detailed logging

## Project Structure

```
SemInd/
├── ga_deap.py              # Main genetic algorithm implementation
├── rfs.py                  # Random Forest Selector baseline
├── requirements.txt        # Dependencies
├── data/                   # Dataset files
├── evaluators/            # Evaluation metrics
├── models/                # ML models
├── preprocess/            # Data preprocessing
├── utils/                 # Utility functions
└── viz/                   # Visualization tools
```

## Important Notes

- **Environment**: Use the provided `requirements.txt` for consistent results
- **Memory**: GA experiments may require significant RAM for large datasets
- **Runtime**: Full experiments (20 seeds × 300 generations) can take several hours
- **Logging**: Results are automatically saved to `logs/` directory
- **Reproducibility**: Always specify seeds for deterministic results

## Hardware Requirements

- **RAM**: ≥8GB for AWA2/CUB, ≥16GB for FLO
- **CPU**: Multi-core recommended for parallel evaluation
- **Storage**: ~1GB for logs and intermediate results

For questions or issues, please open a GitHub issue.