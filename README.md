# SensiBlockFL

PyTorch implementation of layer-sensitivity-guided block-sparse federated communication.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Data and workflow

[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) are downloaded by torchvision. Preparation freezes a class-wise Dirichlet partition; the reserved local probe batch is excluded from local optimization.

```bash
sensiblockfl prepare --config configs/cifar10.json --data data/cifar10
sensiblockfl train --config configs/cifar10.json --data data/cifar10 --output runs/full
sensiblockfl compare --config configs/cifar10.json --data data/cifar10 --output runs/compare
sensiblockfl evaluate --inputs runs/compare/fedavg runs/compare/topk runs/compare/sensiblockfl --output outputs/comparison
sensiblockfl plot --output outputs/comparison
```

For CIFAR-100 and stronger heterogeneity, use the shipped `configs/cifar100-alpha0.5.json`, `configs/cifar100-alpha0.1.json` or `configs/cifar10-alpha0.1.json` with a separate data directory:

```bash
sensiblockfl prepare --config configs/cifar100-alpha0.1.json --data data/cifar100-a01
sensiblockfl train --config configs/cifar100-alpha0.1.json --data data/cifar100-a01 --output runs/cifar100-a01
```

Use `--set alpha=0.1 rate=0.05 seed=1` for numeric overrides. Edit the JSON `device` field to select `cuda:0`; this avoids shell-specific JSON string escaping. `target_accuracy` may be added to the config to set the time-to-target report threshold; its default is 0.5.

## Method and comparisons

Each trainable tensor with dimension greater than one is split into contiguous blocks. The probe score is `abs(dot(gradient, update + residual))/sqrt(parameter_count)`. A client-specific EMA, temperature softmax and capacity-constrained largest-remainder allocation distribute exactly K blocks. Magnitude divided by sqrt(block size) selects blocks inside a layer. One-dimensional parameters are dense. Residuals and EMAs persist across client absences. The server multiplies sample counts by the bounded cosine modifier before normalizing; momentum is used for alignment, rather than as an additional model update.

`--method` selects dense FedAvg, FedProx, SCAFFOLD, coordinate Top-k+EF, random/uniform block+EF, the full method or `no_ef`, `no_sensitivity`, `no_aggregation`. SCAFFOLD uses control-variate update option II with zero local optimizer momentum, resets its optimizer each round and counts uploaded control changes. Other methods use the configured momentum. Bias and BN affine parameters are dense; BN running statistics are sample-weight averaged and integer tracking buffers use maximum. Block IDs cost `ceil(log2(total_blocks)/8)` bytes, at least one; coordinate Top-k uses 32-bit indices. All sparse payloads use FP32 values. Broadcasts, transport headers and running-stat buffers are outside reported uplink payloads.

## Configuration and outputs

The default is CIFAR ResNet-18 with a 3x3 stem, 20 clients, 10 participants, five local epochs, 200 rounds and 10% sparse blocks. `block_size=256`, `beta=0.9`, `temperature=1`, `minimum=1`, `eta=0.1` and server momentum 0.9 are explicit implementation settings. All baselines share the split, seeded client selection, model and optimization budget. Test accuracy is recorded for reporting and never selects a checkpoint. Compare several seeds and upload rates using separate output directories.

`history.csv` contains round accuracy, cumulative uplink bytes, local-training time, compression/probe time and their ratio. `allocations.json` stores the last round's layer budgets. `model.pt` stores the final global model; `config.json` records settings. Evaluation writes comparison CSV/JSON and rounds/bytes to the chosen accuracy threshold; unreachable thresholds are null. `plot` produces PNG/SVG charts from those measured values. The compression time includes the probe and local compression, and device work is synchronized before timing ends. Standard deviation across independent seeds can be computed directly from the per-run histories.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
sensiblockfl-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
