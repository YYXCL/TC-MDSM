# TC-MDSM

Task-conditioned multidimensional state modeling for heterogeneous multi-agent task allocation and scheduling: state encoding, candidate retrieval, graph prediction, CP-SAT scheduling, and dynamic execution simulation.

## Setup

Use Python 3.10 or later. Install a PyTorch build suitable for your platform, then run from the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run

```sh
python evaluate/tc_mdsm_v3_runner.py all --device cpu
```

Use `--device cuda` for a CUDA-enabled PyTorch installation. The default experiment is computationally intensive. Stages can also run separately as `generate`, `train`, `evaluate`, and `report`; use identical experiment arguments for each stage. Pass `--output result/<run-name>` to select a run directory and `--resume` to continue training from its last checkpoint. Existing instance results are skipped during evaluation.

```sh
python evaluate/tc_mdsm_v3_runner.py --help
python evaluate/tc_mdsm_v3_report.py result/experiment
```

## Layout

- `data/`: foundation inputs, source traces, and dataset preparation scripts.
- `model/`: state encoders, predictors, retrieval, training, scheduling, and simulation.
- `evaluate/`: experiment runner and plotting script.
- `tools/`: stage launcher and capability-generation utility.
- `result/`: generated samples, checkpoints, instance results, tables, and figures; excluded from version control.

## Evaluation

Methods include TC-MDSM, GMATANN, HM-MATAS (displayed as TARGETNET), Rule-Latency, Full-State, and StaticGraph. Learning baselines are adapted to this simulation task. All methods use shared scenario and environment seeds. Parameters and request outcomes are saved for each run.

On-time completion rate counts rejection, execution failure, and lateness as failures. Average completion latency includes only requests that succeed before their deadlines; missing latency is not zero. Confidence intervals use seed-level means. Figures are generated from saved simulation results.

The implementation uses task-adapted baselines and does not reproduce every paper setting. It does not provide a hard reliability constraint or failure retries; candidate budgets are configured per run.

See [data/README.md](data/README.md) for input provenance and preparation.
