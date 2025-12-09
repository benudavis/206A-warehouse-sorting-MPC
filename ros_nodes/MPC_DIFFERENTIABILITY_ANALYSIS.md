# MPC Differentiability Analysis: MoveIt IK vs MPC FK

## The Question

**Does MPC have issues with differentiability when using MoveIt IK?**

## Short Answer

**No, there's no differentiability issue** because:
1. MoveIt IK is called **once** (outside the optimization loop) to get target joint angles
2. MPC optimizes in **joint space** (not Cartesian space)
3. MPC uses **differentiable FK** internally for obstacle constraints
4. Target joint angles are **parameters** to the optimization, not variables

## Detailed Analysis

### How MPC Works

```
┌─────────────────────────────────────────────────────────┐
│ MPC Optimization (CasADi/IPOPT)                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Variables: q[k] (joint angles at each time step)       │
│                                                         │
│ Cost Function:                                          │
│   • Tracking: ||q[k] - q_target||²                     │
│   • Smoothness: ||q[k+1] - q[k]||²                     │
│                                                         │
│ Constraints:                                             │
│   • Joint limits: q_min ≤ q[k] ≤ q_max                 │
│   • Velocity limits: |(q[k+1] - q[k])/dt| ≤ v_max      │
│   • Obstacle avoidance: FK(q[k]) outside obstacles     │
│     └─ Uses differentiable FK (CasADi symbolic)        │
│                                                         │
│ Parameters (fixed, not optimized):                     │
│   • q_target (from MoveIt IK) ← NOT in optimization!    │
│   • q_current (current state)                          │
│   • obstacle positions/sizes                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Why MoveIt IK Doesn't Need to be Differentiable

**Key Insight**: MoveIt IK is **NOT part of the optimization loop**

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Get Target (ONCE, before optimization)         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Cartesian target: (x, y, z, quaternion)               │
│         │                                               │
│         ▼                                               │
│ MoveIt IK (black box, not differentiable)              │
│         │                                               │
│         ▼                                               │
│ Joint target: [θ1, θ2, θ3, θ4, θ5, θ6]               │
│         │                                               │
│         └─ This is just a PARAMETER to MPC              │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 2: MPC Optimization (uses differentiable FK)      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ MPC receives:                                           │
│   • q_target = [θ1, θ2, ...] ← Fixed parameter         │
│   • q_current = current joint state                    │
│                                                         │
│ MPC optimizes:                                          │
│   min ||q[k] - q_target||²  (tracking cost)            │
│   + ||q[k+1] - q[k]||²     (smoothness)                │
│   + obstacle_cost(FK(q[k])) (obstacle avoidance)      │
│                                                         │
│ Where FK(q[k]) is:                                      │
│   • Analytic FK (CasADi symbolic)                       │
│   • Fully differentiable                                │
│   • Used for obstacle constraints                       │
│                                                         │
│ MoveIt IK is NOT used here!                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### What Needs to be Differentiable

✅ **Must be differentiable:**
- **FK function** (for obstacle constraints) - ✅ Uses CasADi symbolic FK
- **Cost function** (tracking, smoothness) - ✅ Quadratic, fully differentiable
- **Constraints** (joint limits, velocities) - ✅ Linear/quadratic, differentiable

❌ **Does NOT need to be differentiable:**
- **MoveIt IK** - Only called once to get target, not in optimization
- **Target joint angles** - Just parameters, not optimized variables

### The Flow

```python
# OUTSIDE optimization (happens once):
target_cartesian = [x, y, z, qx, qy, qz, qw]
target_joints = moveit_ik.compute_ik(target_cartesian)  # Black box, OK!

# INSIDE optimization (MPC loop):
def mpc_cost(q_trajectory, q_target, obstacles):
    """
    q_trajectory: optimization variables (joint angles)
    q_target: parameter (from MoveIt IK, fixed)
    obstacles: parameters (fixed)
    """
    cost = 0
    
    # Tracking cost (differentiable in q_trajectory)
    for k in range(horizon):
        cost += ||q_trajectory[k] - q_target||²
    
    # Obstacle cost (uses differentiable FK)
    for k in range(horizon):
        ee_pos = fk(q_trajectory[k])  # ← Differentiable CasADi FK
        cost += obstacle_penalty(ee_pos, obstacles)
    
    return cost  # Fully differentiable in q_trajectory
```

### Why This Works

1. **Separation of Concerns**:
   - **IK**: Converts Cartesian → Joint (happens once, outside optimization)
   - **MPC**: Optimizes joint trajectory (happens repeatedly, uses FK)

2. **MPC Works in Joint Space**:
   - Variables: `q[k]` (joint angles)
   - Target: `q_target` (joint angles from IK)
   - No Cartesian space in optimization!

3. **FK is Only for Obstacles**:
   - MPC uses FK to check if end-effector collides with obstacles
   - FK is differentiable (CasADi symbolic)
   - This enables gradient-based optimization

### Potential Issues (and Why They Don't Matter)

#### Issue 1: "What if MoveIt IK gives a bad solution?"
- **Not a problem**: MPC will still optimize to reach that target
- If target is unreachable, MPC will get as close as possible
- Obstacle constraints will prevent collisions

#### Issue 2: "What if we need to optimize in Cartesian space?"
- **Current approach**: Optimize in joint space (simpler, faster)
- **Alternative**: Could use differentiable IK, but not necessary
- Joint space optimization is standard for MPC

#### Issue 3: "What about IK singularities?"
- MoveIt IK handles singularities (returns error or alternative solution)
- MPC doesn't care - it just optimizes to the target joint angles it receives
- If target is at singularity, MPC will approach it as close as possible

### Code Evidence

From `mpc_controller.py`:

```python
# Line 237: MPC uses differentiable FK
ee_pos_k = self.fk_fun(q[:, k])  # ← CasADi symbolic, differentiable

# Line 352: Target is a parameter (not optimized)
q_target = np.asarray(target_state, dtype=float)  # ← Fixed parameter

# Line 378: Target passed as parameter
params = np.concatenate([
    q_current,      # Parameter
    q_target,       # Parameter (from MoveIt IK)
    obs_pos_flat,   # Parameter
    obs_size_flat,  # Parameter
    n_active_obs,    # Parameter
])
```

### Summary

| Component | Differentiable? | Used in Optimization? | Impact |
|-----------|----------------|---------------------|---------|
| MoveIt IK | ❌ No | ❌ No (called once) | None - just provides target |
| MPC FK | ✅ Yes (CasADi) | ✅ Yes (for obstacles) | Critical - enables gradients |
| Cost function | ✅ Yes (quadratic) | ✅ Yes | Critical - enables optimization |
| Target joints | N/A (parameter) | ❌ No | None - just a fixed value |

## Conclusion

**No differentiability issues!** The architecture is well-designed:

1. ✅ MoveIt IK provides target (outside optimization)
2. ✅ MPC optimizes in joint space (no IK needed in loop)
3. ✅ MPC uses differentiable FK for obstacles (enables gradients)
4. ✅ Everything that needs gradients has them

The separation is clean: **IK for planning, FK for optimization**.
