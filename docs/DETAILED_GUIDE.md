# Detailed Project Guide

## Table of Contents

1. [MPC Controller](#mpc-controller)
2. [Data Collection](#data-collection)
3. [Neural Network Training](#neural-network-training)
4. [Evaluation](#evaluation)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## MPC Controller

### Overview

The MPC controller (`src/control/mpc_controller.py`) implements position-space trajectory optimization for robots with position servos (like the UR5e).

### Formulation

```
minimize:   Σ ||q_k - q_target||²_Q              (position error)
          + Σ ||q_k - q_{k-1}||²_R               (smoothness)
          + ||q_N - q_target||²_{Q_terminal}     (terminal cost)

subject to: q_0 = q_current                      (initial condition)
            |q_{k+1} - q_k|/dt ≤ v_max          (velocity limits)
            q_min ≤ q_k ≤ q_max                 (joint limits)
```

### Tuning Parameters

**Default settings (optimized):**
- `horizon = 30` - Prediction horizon
- `Q = 500` - Position error weight
- `Q_terminal = 1000` - Terminal cost weight
- `R = 0.1` - Smoothness weight
- `max_velocity = 5.0` rad/s

**Tuning guidelines:**
- ↑ Q/Q_terminal → More aggressive tracking
- ↑ R → Smoother motion
- ↑ horizon → Better planning, slower computation
- ↑ max_velocity → Faster motion

### Performance

- **Success rate:** ~100% on reachable targets
- **Convergence:** ~100-300 steps depending on distance
- **Error reduction:** 96%+
- **Computation time:** ~50ms per step (with warm starting)

---

## Data Collection

### Script: `scripts/collect_mpc_data.py`

**Purpose:** Run MPC on random targets and record demonstrations.

**Key observations (32-dimensional):**
- Robot state (12): joint positions + velocities
- End effector pose (7): position + orientation
- Object pose (7): position + orientation  
- Target position (6): desired joint angles

**Actions (6-dimensional):**
- Next desired joint positions from MPC

### Usage

```bash
# Standard collection
uv run python scripts/collect_mpc_data.py \
    --episodes 50 \
    --steps 300

# Quick test
uv run python scripts/collect_mpc_data.py \
    --episodes 10 \
    --steps 100
```

**Output:**
- `data/raw/mpc_data_TIMESTAMP.npz` - Observations and actions
- `data/raw/mpc_metadata_TIMESTAMP.json` - Collection metadata

### Data Quality

Good data collection requires:
- ✅ Diverse target positions
- ✅ MPC successfully reaching most targets
- ✅ Sufficient episode length (300 steps recommended)
- ✅ No solver failures

**Tip:** Check metadata file for success statistics.

---

## Neural Network Training

### Architecture

```
Input (32) → FC(128) → ReLU → LayerNorm →
             FC(128) → ReLU → LayerNorm →
             FC(64)  → ReLU → LayerNorm →
             FC(6)   → Output
```

### Training Process

The network learns:
```
observation → action
(current state + target) → (next position)
```

**Data normalization:**
- Observations: zero mean, unit variance
- Actions: zero mean, unit variance
- Applied during training, inverted during inference

### Usage

```bash
uv run python scripts/train_imitator.py \
    --data data/raw/mpc_data_TIMESTAMP.npz \
    --epochs 150 \
    --batch-size 128 \
    --lr 0.001
```

**Hyperparameters:**
- `--epochs`: Training epochs (150 recommended)
- `--batch-size`: Batch size (64-128 works well)
- `--lr`: Learning rate (0.001 default)
- `--hidden`: Network architecture (e.g., `--hidden 256 256 128`)

### Expected Results

**Good training:**
- Train loss < 0.001 after 50 epochs
- Val loss < 0.001 and stable
- Smooth loss curves

**Red flags:**
- Val loss increasing → Overfitting
- Loss not decreasing → Learning rate too low or bad data
- Oscillating loss → Learning rate too high

---

## Evaluation

### Script: `scripts/evaluate_imitator.py`

Compares MPC (expert) vs learned controller on random targets.

**Metrics:**
- Success rate (% reaching target)
- Average steps to target
- Final position error
- Trajectory plots

### Usage

```bash
uv run python scripts/evaluate_imitator.py \
    --model data/models/mpc_imitator_TIMESTAMP.pth \
    --trials 20
```

**Output:**
- Console: Success rates, average steps, errors
- `data/processed/evaluation_results.png` - Comparison plots

### Interpreting Results

**Success criteria:**
- Position error < 0.05 rad within 400 steps

**Good performance:**
- Learned success rate ≥ 80% of MPC success rate
- Learned steps ≤ 1.5x MPC steps
- Similar error trajectories

---

## Configuration

### File: `config/system_config.yaml`

**MPC settings:**
```yaml
mpc:
  horizon: 30
  dt: 0.01
  weights:
    position_error: 500.0
    terminal: 1000.0
    smoothness: 0.1
```

**Robot settings:**
```yaml
robot:
  n_joints: 6
  joint_limits:
    lower: [-6.28, -6.28, -6.28, -6.28, -6.28, -6.28]
    upper: [6.28, 6.28, 6.28, 6.28, 6.28, 6.28]
  max_velocity: 5.0
```

**Training settings:**
```yaml
training:
  epochs: 150
  batch_size: 128
  learning_rate: 0.001
  hidden_dims: [128, 128, 64]
```

---

## Troubleshooting

### MPC Issues

**Problem:** MPC not converging
- **Solution:** Increase `Q` and `Q_terminal` weights
- **Solution:** Increase horizon
- **Solution:** Check joint limits aren't too restrictive

**Problem:** MPC too slow
- **Solution:** Decrease horizon
- **Solution:** Increase acceptable_tol in solver options
- **Solution:** Use warm starting (already enabled)

### Data Collection Issues

**Problem:** Many solver failures
- **Solution:** Use easier targets (smaller changes)
- **Solution:** Increase max_velocity
- **Solution:** Check simulation timestep

**Problem:** Not enough diversity
- **Solution:** Increase number of episodes
- **Solution:** Widen target distribution bounds

### Training Issues

**Problem:** High training loss
- **Solution:** Train longer (more epochs)
- **Solution:** Increase network size
- **Solution:** Decrease learning rate

**Problem:** Overfitting (high val loss)
- **Solution:** Collect more data
- **Solution:** Add regularization
- **Solution:** Smaller network

**Problem:** Model doesn't work at inference
- **Solution:** Check observation includes target!
- **Solution:** Verify normalization applied correctly
- **Solution:** Check model input dimensions match training

### macOS Specific

**Problem:** `mjpython` library error
```bash
# Solution: Create symlink
mkdir -p .venv/lib
ln -sf ~/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/libpython3.12.dylib .venv/lib/libpython3.12.dylib
```

**Problem:** PyTorch load error
```bash
# Solution: Already fixed in code with weights_only=False
```

---

## Advanced Topics

### Adding Vision

To replace state observations with camera images:

1. Modify `src/perception/sim_state.py` to capture RGB images
2. Change network to CNN architecture
3. Update observation dimension in training

### Real Robot Deployment

To deploy on actual UR7e:

1. Implement `src/control/robot_interface.py` with URX library
2. Test MPC in sim with same control frequency
3. Collect initial data on real robot
4. Fine-tune with real-world data

### Online Learning

To improve model during deployment:

1. Record (observation, MPC action) during execution
2. Periodically retrain model
3. Use DAgger algorithm for iterative improvement

---

## References

**MPC:**
- CasADi documentation: https://web.casadi.org/
- Rawlings, J. B., & Mayne, D. Q. (2009). Model predictive control: Theory and design.

**Imitation Learning:**
- Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning.
- Pomerleau, D. A. (1988). Alvinn: An autonomous land vehicle in a neural network.

**Robot Control:**
- MuJoCo documentation: https://mujoco.readthedocs.io/
- Universal Robots documentation

---

## Contact

For questions about this project, contact the team members through the course staff.

