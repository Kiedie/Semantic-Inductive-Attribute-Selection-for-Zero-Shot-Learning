# SemInd: Semantic Inductive Attribute Selection for Zero-Shot Learning

A unified framework combining genetic algorithms (GA) and semantic inductive methods (RFS) for zero-shot learning attribute selection.

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

Create a virtual environment and install dependencies available in *requirements.txt*


## Quick Start

### 1. Genetic Algorithm (GA) Experiments

Run genetic algorithm optimization on AWA2 dataset:

```bash
python ga_deap.py --dataset AWA2 --generations 300 --population-size 50 --seed 42
```

### 2. Random Forest Selector (RFS) Baseline

Run RFS baseline comparison:

```bash
python ga_deap.py \
    --dataset CUB \
    --generations 200 \
    --population-size 100 \
    --crossover-probability 0.3 \
    --mutation-probability 0.7 \
    --seed 42 \
    --test-eval \
    --test-frequency 20 \
    --cv \
    --n-splits 5 \
    --n-jobs -1 \
    --verbose
```

## Reproducible Experiments

To reproduce the results from our paper, use the exact seeds from our experiments:

### Seeds Used in GA Experiments

For reproducible experiments, use these seeds (20 different seeds for statistical significance). Example:

```bash
SEEDS=(6139 9401 9012 5278 3064 8888 7815 1593 3562 5678 4680 7891 2856 2351 2024 3491 2348 4782 2468 8719)
```



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
- **Logging**: Results are automatically saved to `logs/` directory (it is necessary to create this directory previously)
- **Reproducibility**: Always specify seeds for deterministic results

## Hardware Requirements

- **CPU**: Multi-core recommended for parallel evaluation
- **Storage**: ~1GB for logs and intermediate results

For questions or issues, please open a GitHub issue.