# MPC-Based Robot Control with Imitation Learning

**EECS/BioE/MechE 106A/206A - Fall 2025**

Model Predictive Control for robotic manipulation with analytical kinematics and comprehensive diagnostic tools.

## Team

- Kathy Min - Mechanical Engineering PhD
- Ben Davis - Mechanical Engineering PhD  
- Sharaf Hossain - MEng Mechanical Engineering
- Parham Sharafoleslami - MEng Mechanical Engineering

## Project Overview

This project implements Model Predictive Control (MPC) for a UR5e robotic arm with analytical kinematics and real-time obstacle avoidance.

**Key Components:**
- ✅ Position-space MPC using CasADi optimization
- ✅ Analytical forward kinematics (DH parameters)
- ✅ Inverse kinematics solver for reach planning
- ✅ Real-time receding horizon control
- ✅ Warehouse color-sorting demonstration
- ✅ Comprehensive diagnostic and test suite

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
Sort red and blue boxes into separate baskets. MPC uses real-time receding horizon control with waypoint guidance to navigate around obstacles.

### Run Tests

Verify kinematics implementation:

```bash
# Run all tests
uv run python tests/run_all_tests.py

# Run individual tests
uv run python tests/test_forward_kinematics.py
uv run python tests/test_inverse_kinematics.py
```

See [tests/README.md](tests/README.md) for test documentation.

---

## Features

- **MPC Controller**: Real-time receding horizon control with CasADi/IPOPT
- **Forward Kinematics**: Analytical FK using DH parameters
- **Inverse Kinematics**: Damped least squares IK solver
- **Obstacle Avoidance**: MuJoCo-based collision detection
- **Diagnostics**: Comprehensive logging and visualizations
- **Test Suite**: Automated FK/IK validation

See [docs/GUIDE.md](docs/GUIDE.md) for detailed documentation.

---

## Project Structure

```
├── scripts/          # Demonstration scripts
│   ├── demo_sorting.py
│   └── demo_color_sorting.py
├── tests/            # Test suite
│   ├── test_forward_kinematics.py
│   ├── test_inverse_kinematics.py
│   └── run_all_tests.py
├── src/              # Source code
│   ├── control/      # MPC, FK, IK controllers
│   ├── diagnostics/  # Logging utilities
│   ├── learning/     # Imitation learning
│   └── perception/   # State estimation
├── sim/models/       # MuJoCo URDF/XML models
├── config/           # System configuration
└── data/             # Generated data
```

## Documentation

- **README.md** (this file) - Quick start
- **[docs/GUIDE.md](docs/GUIDE.md)** - Complete documentation

## Requirements

Python 3.12+, MuJoCo 3.3.6, PyTorch, CasADi, NumPy

See `pyproject.toml` for full dependencies.

## License

Academic project - UC Berkeley EECS Department
