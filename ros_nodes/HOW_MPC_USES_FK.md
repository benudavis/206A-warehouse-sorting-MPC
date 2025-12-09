# How MPC Uses FK for Horizon 10 Path Generation

## The Question

**When MPC generates a path with horizon 10, does it use FK? How?**

## Answer: Yes! FK is used at EVERY step of the horizon

### Visual Overview

```
MPC Optimization (Horizon H=10, so 11 steps: k=0 to k=10)

┌─────────────────────────────────────────────────────────────┐
│ Optimization Variables:                                      │
│   q[0], q[1], q[2], ..., q[10]  (11 joint configurations)   │
└─────────────────────────────────────────────────────────────┘

For EACH step k in [0, 1, 2, ..., 10]:
    ┌─────────────────────────────────────────────────────┐
    │ Step k:                                               │
    │   1. Get joint angles: q[k] = [θ1, θ2, θ3, θ4, θ5, θ6] │
    │                                                       │
    │   2. Compute end-effector position using FK:         │
    │      ee_pos[k] = FK(q[k])                            │
    │      └─ This is a CasADi symbolic expression!        │
    │                                                       │
    │   3. For EACH obstacle:                              │
    │      • Check if ee_pos[k] is inside obstacle         │
    │      • Add HARD CONSTRAINT: ee_pos[k] must be OUT    │
    │      • Add SOFT COST: penalty if too close           │
    │                                                       │
    └─────────────────────────────────────────────────────┘
```

## Code Breakdown

### From `mpc_controller.py` lines 235-272:

```python
# For horizon H=10, this loop runs 11 times (k=0 to k=10)
for k in range(H + 1):  # H+1 = 11 steps
    # Step 1: Compute end-effector position using FK
    # q[:, k] = joint angles at step k (6 values)
    # ee_pos_k = end-effector position at step k (3 values: x, y, z)
    ee_pos_k = self.fk_fun(q[:, k])  # ← FK CALL HERE!
    
    # Step 2: For each obstacle, check collision
    for i_obs in range(self.n_max_obstacles):
        center = obs_pos[:, i_obs]      # Obstacle center [x, y, z]
        half_size = obs_size[:, i_obs]  # Obstacle half-size [sx, sy, sz]
        
        # Compute distance from EE to obstacle center
        diff = ee_pos_k - center  # [dx, dy, dz]
        
        # HARD CONSTRAINT: EE must be OUTSIDE obstacle
        # (This becomes a constraint in the optimization)
        inside_clearances = half_size - ca.fabs(diff)
        min_inside = ca.fmin(...)  # Minimum clearance
        constraints.append(min_inside <= 0)  # Must be <= 0 (outside)
        
        # SOFT COST: Penalty for being too close
        # (This adds to the cost function)
        inflated = half_size + safety_margin
        u_margin = ca.fabs(diff) - inflated
        cost += obstacle_weight * penalty(u_margin)
```

## Detailed Example: Horizon 10 with 2 Obstacles

### What Happens:

```
Horizon H = 10 → 11 steps (k = 0, 1, 2, ..., 10)
Number of obstacles = 2

Total FK calls: 11 (one per step)
Total obstacle checks: 11 × 2 = 22

┌─────┬──────────────────────────────────────────────────────────┐
│  k  │ What MPC Does                                            │
├─────┼──────────────────────────────────────────────────────────┤
│  0  │ q[0] → FK → ee_pos[0] → Check vs Obstacle1, Obstacle2  │
│  1  │ q[1] → FK → ee_pos[1] → Check vs Obstacle1, Obstacle2  │
│  2  │ q[2] → FK → ee_pos[2] → Check vs Obstacle1, Obstacle2  │
│ ... │ ...                                                      │
│ 10  │ q[10] → FK → ee_pos[10] → Check vs Obstacle1, Obstacle2│
└─────┴──────────────────────────────────────────────────────────┘
```

### The Optimization Problem:

```
Minimize:
  Cost = Tracking + Smoothness + Obstacle_Penalty
  
Where:
  Tracking = Σ ||q[k] - q_target||²  (for k=0 to 9)
  Smoothness = Σ ||q[k+1] - q[k]||²  (for k=0 to 9)
  
  Obstacle_Penalty = Σ penalty(ee_pos[k], obstacles)  (for k=0 to 10)
                     └─ Uses FK: ee_pos[k] = FK(q[k])
  
Subject to:
  • q[0] = q_current  (initial condition)
  • Joint limits: q_min ≤ q[k] ≤ q_max  (for all k)
  • Velocity limits: |(q[k+1] - q[k])/dt| ≤ v_max  (for all k)
  • Obstacle constraints: ee_pos[k] outside obstacles  (for all k)
                          └─ Uses FK: ee_pos[k] = FK(q[k])
```

## Why FK is Needed

### The Problem:
- **MPC optimizes in joint space**: Variables are `q[k]` (joint angles)
- **Obstacles are in Cartesian space**: Obstacles have positions `[x, y, z]`
- **Need to check collisions**: End-effector must avoid obstacles

### The Solution:
- **FK converts joint space → Cartesian space**: `FK(q[k]) → ee_pos[k]`
- **Check collision in Cartesian space**: `ee_pos[k]` vs `obstacle_center`
- **FK must be differentiable**: So optimizer can compute gradients

## Step-by-Step: What MPC Does

### 1. Setup (happens once, when MPC is initialized):

```python
# Build FK function (CasADi symbolic)
self.fk_fun = build_ur5e_fk_function()
# fk_fun: q (6,) → ee_pos (3,)
# This is a CasADi Function, fully symbolic and differentiable
```

### 2. Optimization Setup (happens once, when MPC is initialized):

```python
# Create symbolic variables for optimization
q = ca.SX.sym("q", n_joints, H + 1)  # 6 × 11 matrix

# For each step k in the horizon:
for k in range(H + 1):  # k = 0, 1, 2, ..., 10
    # Compute EE position symbolically
    ee_pos_k = self.fk_fun(q[:, k])  # Symbolic expression!
    
    # Add constraints and costs using ee_pos_k
    # (This builds the optimization problem, doesn't solve it yet)
```

### 3. Optimization Solve (happens every MPC step):

```python
# MPC receives:
#   - q_current: current joint state
#   - q_target: target joint state (from MoveIt IK)
#   - obstacles: list of obstacle positions/sizes

# MPC solves:
#   Find q[0], q[1], ..., q[10] that:
#     • Minimize cost (tracking + smoothness + obstacle penalty)
#     • Satisfy constraints (joint limits, velocity, obstacles)
#
#   During optimization, CasADi automatically:
#     • Evaluates FK(q[k]) for each candidate solution
#     • Computes gradients of FK (for optimization)
#     • Checks obstacle constraints
#     • Updates q[k] values to find optimal trajectory
```

## Visual Example: Horizon 10 Trajectory

```
Time step:  0    1    2    3    4    5    6    7    8    9   10
            │    │    │    │    │    │    │    │    │    │    │
Joint:    q[0] q[1] q[2] q[3] q[4] q[5] q[6] q[7] q[8] q[9] q[10]
            │    │    │    │    │    │    │    │    │    │    │
            ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
FK:      FK() FK() FK() FK() FK() FK() FK() FK() FK() FK() FK()
            │    │    │    │    │    │    │    │    │    │    │
            ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
EE Pos:  p[0] p[1] p[2] p[3] p[4] p[5] p[6] p[7] p[8] p[9] p[10]
            │    │    │    │    │    │    │    │    │    │    │
            ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
Check:   vs   vs   vs   vs   vs   vs   vs   vs   vs   vs   vs
         obs  obs  obs  obs  obs  obs  obs  obs  obs  obs  obs
```

## Key Points

1. **FK is called symbolically**: Not evaluated numerically during setup, but built as a symbolic expression
2. **FK is evaluated during optimization**: CasADi evaluates FK(q[k]) for each candidate solution
3. **FK is used for constraints**: Hard constraint that EE must be outside obstacles
4. **FK is used for costs**: Soft penalty if EE gets too close to obstacles
5. **FK is differentiable**: CasADi automatically computes gradients for optimization

## Why This Matters

- **Without FK**: MPC can't check if end-effector collides with obstacles
- **With FK**: MPC can ensure the entire predicted trajectory avoids obstacles
- **Differentiable FK**: Enables gradient-based optimization (fast convergence)

## Summary

**Yes, MPC uses FK at every step of the horizon!**

For horizon 10:
- 11 FK calls (one per step k=0 to k=10)
- Each FK converts joint angles → end-effector position
- Each end-effector position is checked against all obstacles
- This ensures the entire predicted trajectory avoids obstacles

The FK is:
- **Symbolic** (CasADi): Built once, used many times
- **Differentiable**: Enables gradient-based optimization
- **Efficient**: CasADi optimizes the computation
