# SensiBlockFL

Implementation of layer-sensitivity-guided block-sparse federated communication.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision, freezes a Dirichlet client partition and reserves local probe batches. Use one prepared directory across methods.

```bash
sensiblockfl prepare --config configs/cifar10.json --data data/cifar10
```

## Training and execution

`configs/cifar10.json` records seed `0`. Keep it fixed across paired methods and use separate output directories.

```bash
sensiblockfl train --config configs/cifar10.json --data data/cifar10 --output runs/full
```

## Evaluation

```bash
sensiblockfl evaluate --config configs/cifar10.json --inputs runs/full --output outputs/main_evaluation
```

## Comparisons and ablations

`--method` selects the implemented method or component control. Train separate checkpoints where applicable. `compare` fits the configured methods on the same client partition; evaluate their resulting directories together.

```bash
sensiblockfl compare --config configs/cifar10.json --data data/cifar10 --output runs/compare
sensiblockfl evaluate --config configs/cifar10.json --inputs runs/compare/fedavg runs/compare/topk runs/compare/sensiblockfl --output outputs/comparison
sensiblockfl plot --output outputs/comparison
```

## Outputs

The explicit run directories contain `model.pt`, `metrics.json`, `history.csv`, `allocations.json`, `time_to_target.json`. Comparisons write CSV/JSON and plots derived from those saved measurements.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
