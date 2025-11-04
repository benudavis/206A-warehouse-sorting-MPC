# MPC-Based Robot Control with Imitation Learning

**EECS/BioE/MechE 106A/206A - Fall 2025**

Model Predictive Control for robotic manipulation with neural network imitation learning and comprehensive diagnostic tools.

## Team

- Kathy Min - Mechanical Engineering PhD
- Ben Davis - Mechanical Engineering PhD  
- Sharaf Hossain - MEng Mechanical Engineering
- Parham Sharafoleslami - MEng Mechanical Engineering

## Project Overview

This project implements Model Predictive Control (MPC) for a UR5e robotic arm and trains neural networks to imitate MPC behavior through behavioral cloning.

**Key Components:**
- ✅ Position-space MPC using CasADi optimization
- ✅ Inverse kinematics for reach planning
- ✅ Neural network for behavior cloning
- ✅ Complete data collection and training pipeline
- ✅ Warehouse-style object sorting demonstration
- ✅ Comprehensive diagnostic and logging framework

---

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# macOS only: Create symlink for mjpython
mkdir -p .venv/lib
ln -sf ~/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/libpython3.12.dylib .venv/lib/libpython3.12.dylib
```

### Run Demos

```bash
# Basic warehouse sorting demo
uv run mjpython scripts/demo_sorting.py

# Sorting with comprehensive diagnostics (recommended for debugging)
uv run mjpython scripts/demo_sorting_with_diagnostics.py

# Shelf sorting demo
uv run mjpython scripts/demo_shelf_sorting.py

# MPC vs learned controller comparison
uv run mjpython scripts/demo_comparison.py --mode mpc
```

---

## MPC Imitation Learning Pipeline

Complete workflow from data collection to evaluation:

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

# 4. Visual comparison (MPC vs learned)
uv run mjpython scripts/demo_comparison.py \
    --mode learned \
    --model data/models/mpc_imitator_TIMESTAMP.pth
```

---

## Diagnostic Framework

Comprehensive logging and visualization for debugging:

```python
from src.diagnostics import DiagnosticLogger

logger = DiagnosticLogger(model, data, site_name="arm_hand_pinch")
logger.add_tracked_object("red_box", body_id, size=0.02)
logger.log_state("red_box", "approach", attempt=0)
logger.generate_report(output_dir="data/diagnostics")
```

Generates 12-subplot visualizations, metrics, and failure analysis. See `scripts/demo_sorting_with_diagnostics.py` and `docs/DETAILED_GUIDE.md`.

---

## Project Structure

```
├── src/
│   ├── control/
│   │   ├── mpc_controller.py        # MPC implementation
│   │   └── inverse_kinematics.py    # IK solver
│   ├── learning/
│   │   └── mpc_imitator.py          # Neural network
│   ├── perception/
│   │   └── sim_state.py             # State extraction
│   └── diagnostics/                  # Diagnostic framework
│       ├── logger.py                 # Data logging
│       ├── plotter.py                # Visualizations
│       └── metrics.py                # Performance metrics
├── scripts/
│   ├── demo_sorting.py               # Basic sorting demo
│   ├── demo_sorting_with_diagnostics.py  # Sorting with logging
│   ├── demo_shelf_sorting.py         # Shelf sorting demo
│   ├── collect_mpc_data.py           # Data collection
│   ├── train_imitator.py             # Neural network training
│   ├── evaluate_imitator.py          # Performance evaluation
│   └── demo_comparison.py            # MPC vs learned comparison
├── config/
│   └── system_config.yaml            # System configuration
├── data/
│   ├── raw/                          # Collected MPC data
│   ├── models/                       # Trained models
│   ├── processed/                    # Evaluation results
│   └── diagnostics/                  # Diagnostic outputs
├── docs/
│   └── DETAILED_GUIDE.md             # Technical documentation
└── sim/
    └── models/                       # Robot/gripper models
```

---

## Results

### MPC Performance
- Successfully reaches targets in ~100-300 steps
- 96%+ error reduction
- Smooth, optimal trajectories
- Real-time control at 100Hz simulation rate

### Imitation Learning
- 15,000+ expert demonstrations collected
- Neural network training converges (loss < 0.0001)
- Complete pipeline from collection to deployment
- Evaluation framework with metrics

### Diagnostic Framework
- Automatic data collection (robot state, object poses, distances)
- 12-subplot comprehensive visualizations
- Performance metrics (IK/MPC convergence rates, timing)
- Failure mode identification (lift tests, distance analysis)

---

## Key Features

### 1. Model Predictive Control
- Position-space control using CasADi optimization
- Configurable horizon (default: 30 steps)
- Quadratic cost function with terminal weight
- Handles joint limits and constraints

### 2. Inverse Kinematics
- Damped least squares solver
- Orientation control (quaternions)
- Configurable tolerance and max iterations
- Robust to singular configurations

### 3. Imitation Learning
- Behavioral cloning from MPC demonstrations
- 3-layer feedforward network
- State: joint pos/vel (12D) → Action: joint targets (6D)
- PyTorch implementation with GPU support

### 4. Diagnostic Tools
- Comprehensive data logging
- Rich visualizations (3D trajectories, phase analysis)
- Performance metrics and statistics
- Failure mode identification
- Modular and reusable across demos

---

## Documentation

- **README.md** (this file) - Quick start and overview
- **docs/DETAILED_GUIDE.md** - Technical details, MPC formulation, diagnostics, troubleshooting

---

## Requirements

- Python 3.12+
- MuJoCo 3.3.6
- PyTorch 2.0+
- CasADi 3.6+
- NumPy, Matplotlib, SciPy

See `pyproject.toml` for complete dependencies.

---

## Troubleshooting

### MuJoCo Viewer Issues
- Ensure `mjpython` is used for scripts with visualization
- On macOS, verify libpython symlink exists in `.venv/lib/`

### MPC Not Converging
- Increase horizon length (trade-off: slower)
- Tune position weight (default: 500.0)
- Check target is reachable
- See `docs/DETAILED_GUIDE.md` for tuning guide

### IK Failing
- Increase max iterations (default: 500)
- Relax tolerance (default: 0.01)
- Check target is within workspace
- Verify orientation quaternion is normalized

### Diagnostic Plots Not Showing
- Ensure matplotlib backend is configured
- Use `show_plots=True` in `generate_report()`
- Check output directory permissions

---

## License

Academic project - UC Berkeley EECS Department

---

## Citation

If you use this code in your research, please cite:

```
@misc{warehouse-sorting-mpc-2025,
  title={MPC-Based Robot Control with Imitation Learning},
  author={Min, Kathy and Davis, Ben and Hossain, Sharaf and Sharafoleslami, Parham},
  year={2025},
  institution={UC Berkeley EECS Department}
}
```
