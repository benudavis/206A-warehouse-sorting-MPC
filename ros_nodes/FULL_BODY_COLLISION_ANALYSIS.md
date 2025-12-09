# Full-Body Collision Checking in MPC

## Current State: Only End-Effector is Checked

### What MPC Currently Does

```
┌─────────────────────────────────────────────────────────┐
│ MPC Optimization (CasADi)                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ For each step k in horizon:                           │
│   • q[k] → FK → ee_pos[k]                             │
│   • Check: ee_pos[k] outside obstacles                │
│                                                         │
│ ❌ Only end-effector is checked!                       │
│ ❌ Shoulder, upper arm, forearm, wrist are NOT checked │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MuJoCo Full-Body Check (Heuristic Only)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ • Checks all links: shoulder, upper arm, forearm, etc. │
│ • Used ONLY for initial guess generation               │
│ • NOT used in optimization constraints                 │
│ • NOT differentiable                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Code Evidence

**From `mpc_controller.py` lines 233-272:**
```python
# Only EE is checked in optimization
for k in range(H + 1):
    ee_pos_k = self.fk_fun(q[:, k])  # ← Only EE!
    
    for i_obs in range(self.n_max_obstacles):
        # Check if ee_pos_k collides with obstacle
        constraints.append(ee_pos_k outside obstacle)
```

**From `mpc_controller.py` lines 424-468:**
```python
# Full-body check exists but is ONLY used as heuristic
def _trajectory_collides(self, trajectory, model, data_scratch, site_id):
    # Checks EE + all link bodies
    for bid in self.link_body_ids:  # shoulder, upper arm, etc.
        points.append(data_scratch.xpos[bid].copy())
    
    # But this is ONLY called for initial guess (line 391)
    # NOT in the optimization constraints!
```

## The Problem

**Current limitation:**
- ✅ End-effector avoids obstacles (in optimization)
- ❌ Shoulder, upper arm, forearm, wrist can collide (not checked in optimization)
- ⚠️ Full-body check exists but only as heuristic (not guaranteed)

**Why this matters:**
- Robot arm links can be large (especially upper arm and forearm)
- Obstacles can be between base and end-effector
- Only checking EE means links can pass through obstacles

## Solutions

### Option 1: Add FK for Key Points (Recommended)

**Approach:** Compute FK for multiple critical points on the robot arm.

**What we need:**
- FK for shoulder position
- FK for elbow position  
- FK for wrist position
- FK for end-effector (already have)

**Implementation:**
```python
# In forward_kinematics.py
def build_ur5e_fk_multipoint_function():
    """
    Returns FK functions for multiple key points:
    - shoulder_pos(q) → position
    - elbow_pos(q) → position
    - wrist_pos(q) → position
    - ee_pos(q) → position (already have)
    """
    # Build FK for each key point
    # All CasADi symbolic, fully differentiable
```

**In MPC optimization:**
```python
for k in range(H + 1):
    # Get positions of all key points
    shoulder_pos = fk_shoulder(q[:, k])
    elbow_pos = fk_elbow(q[:, k])
    wrist_pos = fk_wrist(q[:, k])
    ee_pos = fk_ee(q[:, k])
    
    # Check ALL points against obstacles
    for point in [shoulder_pos, elbow_pos, wrist_pos, ee_pos]:
        for i_obs in range(self.n_max_obstacles):
            constraints.append(point outside obstacle)
```

**Pros:**
- ✅ Fully differentiable (CasADi symbolic)
- ✅ Works in optimization constraints
- ✅ Guarantees collision avoidance
- ✅ Computationally efficient

**Cons:**
- ⚠️ Need to implement FK for multiple points
- ⚠️ May miss collisions between key points (but can add more points)

### Option 2: Approximate Link Geometry

**Approach:** Model each link as a sphere or capsule, compute center positions via FK.

**Implementation:**
```python
# Model links as spheres
link_geometries = {
    'shoulder': {'radius': 0.05, 'fk': fk_shoulder},
    'upper_arm': {'radius': 0.08, 'fk': fk_upper_arm_center},
    'forearm': {'radius': 0.06, 'fk': fk_forearm_center},
    'wrist': {'radius': 0.04, 'fk': fk_wrist},
    'ee': {'radius': 0.05, 'fk': fk_ee},
}

for k in range(H + 1):
    for link_name, link_geom in link_geometries.items():
        link_center = link_geom['fk'](q[:, k])
        link_radius = link_geom['radius']
        
        for i_obs in range(self.n_max_obstacles):
            # Check if sphere (link) intersects with box (obstacle)
            constraints.append(sphere_box_separation(link_center, link_radius, obstacle))
```

**Pros:**
- ✅ Captures full link geometry
- ✅ Differentiable (if FK is differentiable)
- ✅ Can model complex shapes (capsules)

**Cons:**
- ⚠️ Need FK for link centers
- ⚠️ Approximation (spheres may not match exact geometry)

### Option 3: Hybrid Approach (Current + Enhancement)

**Approach:** Keep EE constraints in optimization, add full-body check as post-processing.

**Implementation:**
```python
# In optimization: EE constraints (fast, differentiable)
for k in range(H + 1):
    ee_pos_k = self.fk_fun(q[:, k])
    constraints.append(ee_pos_k outside obstacles)

# After optimization: Full-body validation
q_opt = solver_result.reshape(H + 1, n_joints)
for k in range(H + 1):
    if self._check_full_body_collision(q_opt[k], model, data):
        # Re-optimize with tighter constraints or reject solution
        pass
```

**Pros:**
- ✅ Fast optimization (only EE)
- ✅ Full-body safety check

**Cons:**
- ❌ Not guaranteed (may need to re-optimize)
- ❌ Full-body check not differentiable

### Option 4: Use MoveIt Collision Checking (Not Recommended)

**Approach:** Call MoveIt's collision checker during optimization.

**Why not recommended:**
- ❌ Not differentiable (black box)
- ❌ Slow (service calls during optimization)
- ❌ Breaks CasADi optimization flow

## Recommended Solution: Option 1 (Multi-Point FK)

### Implementation Steps

1. **Extend `forward_kinematics.py`:**
   ```python
   def build_ur5e_fk_multipoint_function():
       """
       Returns a dict of FK functions:
       {
           'shoulder': fk_shoulder(q) → (3,),
           'elbow': fk_elbow(q) → (3,),
           'wrist': fk_wrist(q) → (3,),
           'ee': fk_ee(q) → (3,),
       }
       """
   ```

2. **Update `mpc_controller.py`:**
   ```python
   # Initialize multiple FK functions
   self.fk_functions = {
       'shoulder': build_fk_shoulder(),
       'elbow': build_fk_elbow(),
       'wrist': build_fk_wrist(),
       'ee': build_fk_ee(),
   }
   
   # In optimization:
   for k in range(H + 1):
       for point_name, fk_fun in self.fk_functions.items():
           point_pos = fk_fun(q[:, k])
           # Add constraints for this point
   ```

3. **Key Points to Compute:**
   - **Shoulder:** Base of upper arm (joint 2)
   - **Elbow:** Between upper arm and forearm (joint 4)
   - **Wrist:** Base of wrist (joint 6)
   - **EE:** End-effector (already have)

### Why This Works

- **Differentiable:** All FK functions are CasADi symbolic
- **Efficient:** Computed in parallel during optimization
- **Guaranteed:** Hard constraints ensure no collisions
- **Extensible:** Can add more points if needed

## Current Workaround

Until full-body FK is implemented, the current system:
- ✅ Checks EE in optimization (prevents EE collisions)
- ⚠️ Uses full-body heuristic for initial guess (helps but not guaranteed)
- ❌ Does not guarantee full-body collision avoidance

**Recommendation:** Implement Option 1 (multi-point FK) for proper full-body collision avoidance.
