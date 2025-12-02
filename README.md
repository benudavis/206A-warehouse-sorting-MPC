# MPC-Based Robot Control with Imitation Learning

**EECS/BioE/MechE 106A/206A - Fall 2025**

Model Predictive Control for robotic manipulation with neural network imitation learning and comprehensive diagnostic tools.

## Team

- Kathy Min - Mechanical Engineering PhD
- Ben Davis - Mechanical Engineering PhD  
- Sharaf Hossain - MEng Mechanical Engineering
- Parham Sharafoleslami - MEng Mechanical Engineering

## Project Overview

This project implements Model Predictive Control (MPC) for a UR5e robotic arm with neural network-based obstacle avoidance and imitation learning capabilities.

**Key Components:**
- ✅ Position-space MPC using CasADi optimization
- ✅ Neural network FK for real-time obstacle avoidance
- ✅ Inverse kinematics for reach planning
- ✅ Shelf-based warehouse sorting demonstration
- ✅ Comprehensive diagnostic and logging utilities

---

## Quick Start

### Installation

```bash
uv sync

# macOS: Create symlink for mjpython
mkdir -p .venv/lib
ln -sf ~/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/libpython3.12.dylib .venv/lib/libpython3.12.dylib
```

### Run Demos

**Shelf Stacking:**
```bash
uv run mjpython scripts/demo_sorting.py
```
The UR5e picks three cubes and stacks them on a shelf.

**Color Sorting (with obstacles):**
```bash
uv run mjpython scripts/demo_color_sorting.py
```
Sort red and blue boxes into separate baskets. Obstacles block direct paths, forcing MPC to plan intelligent trajectories using NN FK obstacle avoidance.

### Optional: NN FK for Obstacle Avoidance

One-time setup for advanced obstacle avoidance:

```bash
# Full setup with visualization (5-10 min)
uv run python tools/nn_fk/setup.py

# Quick mode (30 sec)
uv run python tools/nn_fk/setup.py --quick

# See performance diagrams
uv run python tools/nn_fk/visualize_performance.py
open data/diagnostics/nn_fk_performance.png
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training details and [docs/GUIDE.md](docs/GUIDE.md) for full documentation.

---

## Features

- **MPC Controller**: Position-space control with NN FK obstacle avoidance
- **Inverse Kinematics**: Damped least squares IK solver
- **Diagnostics**: Comprehensive logging and 12-subplot visualizations
- **Imitation Learning**: Neural network learns from MPC demonstrations

See [docs/GUIDE.md](docs/GUIDE.md) for detailed documentation.

---

## Project Structure

```
├── scripts/          # Demos
│   └── demo_sorting.py
├── tools/            # Training utilities
│   └── nn_fk/        # NN FK setup scripts
├── src/              # Source code
│   ├── control/      # MPC, IK, NN FK
│   ├── diagnostics/  # Logging
│   ├── learning/     # Imitation learning
│   └── perception/   # State estimation
├── sim/models/       # MuJoCo models
├── config/           # Configuration
├── data/             # Generated data
└── docs/             # Documentation
    └── GUIDE.md
```

## Documentation

- **README.md** (this file) - Quick start
- **[docs/GUIDE.md](docs/GUIDE.md)** - Complete documentation

## Requirements

Python 3.12+, MuJoCo 3.3.6, PyTorch, CasADi, NumPy

See `pyproject.toml` for full dependencies.

## License

Academic project - UC Berkeley EECS Department
