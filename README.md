# MPC-Based Robot Control with Imitation Learning

**EECS/BioE/MechE 106A/206A - Fall 2025**

Model Predictive Control for robotic manipulation with neural network imitation learning.

## Team

- Kathy Min - Mechanical Engineering PhD
- Ben Davis - Mechanical Engineering PhD  
- Sharaf Hossain - MEng Mechanical Engineering
- Parham Sharafoleslami - MEng Mechanical Engineering

## Project Overview

This project implements Model Predictive Control (MPC) for a UR5e robotic arm and explores learning to imitate MPC behavior using neural networks.

**Key Components:**
- Position-space MPC using CasADi optimization
- Neural network for behavior cloning
- Complete data collection and training pipeline
- Evaluation framework comparing MPC vs learned controller

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# macOS only: Create symlink for mjpython
mkdir -p .venv/lib
ln -sf ~/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/libpython3.12.dylib .venv/lib/libpython3.12.dylib
```

### Run MPC Demo

```bash
# Watch MPC controller in action
uv run mjpython scripts/demo_comparison.py --mode mpc
```

## Complete Workflow

```bash
# 1. Collect MPC demonstrations
uv run python scripts/collect_mpc_data.py --episodes 50 --steps 300

# 2. Train neural network
uv run python scripts/train_imitator.py \
    --data data/raw/mpc_data_TIMESTAMP.npz \
    --epochs 150

# 3. Evaluate performance
uv run python scripts/evaluate_imitator.py \
    --model data/models/mpc_imitator_TIMESTAMP.pth \
    --trials 20

# 4. Visual demo
uv run mjpython scripts/demo_comparison.py \
    --mode learned \
    --model data/models/mpc_imitator_TIMESTAMP.pth
```

## Project Structure

```
├── src/
│   ├── control/          # MPC controller
│   ├── learning/         # Neural network
│   └── perception/       # State extraction
├── scripts/              # Workflow scripts
│   ├── collect_mpc_data.py
│   ├── train_imitator.py
│   ├── evaluate_imitator.py
│   └── demo_comparison.py
├── config/               # Configuration
├── data/                 # Data and models
│   ├── raw/             # Collected data
│   ├── models/          # Trained models
│   └── processed/       # Results
└── docs/                 # Documentation
```

## Results

**MPC Performance:**
- Successfully reaches targets in ~100-300 steps
- 96%+ error reduction
- Smooth, optimal trajectories

**Imitation Learning:**
- 15,000 expert demonstrations collected
- Neural network training converges (loss < 0.0001)
- Framework demonstrates complete pipeline

## Documentation

See `docs/DETAILED_GUIDE.md` for:
- MPC formulation and tuning
- Data collection details
- Training procedures
- Troubleshooting

## Requirements

- Python 3.12+
- MuJoCo 3.3.6
- PyTorch 2.0+
- CasADi 3.6+

See `pyproject.toml` for complete dependencies.

## License

Academic project - UC Berkeley EECS Department
