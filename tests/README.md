# Test Suite

Comprehensive tests for Forward Kinematics (FK), Inverse Kinematics (IK), and Model Predictive Control (MPC).

## Overview

These tests verify that the control implementations work correctly and are ready for production use.

## Test Files

### `test_forward_kinematics.py`
Tests the FK implementation (MuJoCo-based):
- **Home Position Test**: Verifies FK at zero configuration
- **Accuracy Test**: Tests 100 random joint configurations
- **Multi-Link Test**: Verifies FK for shoulder, elbow, wrist, and EE

**Pass Criteria:**
- All errors < 2mm: ✅ Excellent
- Max error < 5mm: ⚠️ Acceptable
- Max error > 5mm: ❌ Failed (fix required)

### `test_inverse_kinematics.py`
Tests the IK solver:
- **Convergence Test**: Checks iteration requirements
- **Roundtrip Test**: FK → IK → FK (should return to same position)
- **Workspace Test**: Tests typical positions in workspace

**Pass Criteria:**
- Success rate > 90%: ✅ Good
- Mean error < 20mm: ✅ Acceptable
- Max error < 100mm on difficult poses: ✅ Expected

### `test_mpc_controller.py`
Tests the MPC controller:
- **Basic Solve**: Verifies MPC can find solutions
- **Joint Limits**: Ensures joint bounds are respected
- **Velocity Limits**: Verifies velocity constraints
- **Convergence**: Tests receding horizon drives to target
- **Warm Starting**: Benchmarks warm start performance
- **Dynamic Safety Margin**: Tests runtime margin changes
- **Multiple Obstacles**: Tests obstacle management

**Pass Criteria:**
- All solves succeed: ✅ Required
- Constraints satisfied: ✅ Required
- Convergence > 90% error reduction: ✅ Good
- Solve time < 100ms: ✅ Real-time capable
- Solve time < 200ms: ⚠️ Acceptable

### `run_all_tests.py`
Master test runner that executes all test suites and provides a summary.

## Running Tests

### Run All Tests
```bash
uv run python tests/run_all_tests.py
```

### Run Individual Tests
```bash
# FK tests only
uv run python tests/test_forward_kinematics.py

# IK tests only
uv run python tests/test_inverse_kinematics.py

# MPC tests only
uv run python tests/test_mpc_controller.py
```

## Expected Output

### Successful Run
```
🎉 ALL TESTS PASSED! Kinematics implementation is ready.

✅ PASS: Forward Kinematics
✅ PASS: Inverse Kinematics  
✅ PASS: Model Predictive Control

Results: 3/3 test suites passed
```

### Failed Run
```
⚠️  SOME TESTS FAILED. Review and fix before using in production.

❌ FAIL: Forward Kinematics
  Max error: 350.5 mm (expected < 2mm)
  
✅ PASS: Inverse Kinematics
✅ PASS: Model Predictive Control

Results: 2/3 test suites passed
```

## Debugging Failed Tests

### FK Failures
Common issues:
1. **Wrong DH parameters**: Check against UR5e datasheet
2. **DH convention mismatch**: Verify Modified vs Standard DH
3. **Missing transforms**: Check base offset and tool offset
4. **Joint order**: Ensure q[0..5] maps correctly to joints 1-6

### IK Failures
Common issues:
1. **Poor initialization**: Start from current pose instead of zeros
2. **Small step size**: Increase damping or learning rate
3. **Wrong Jacobian**: Verify FK is correct first
4. **Tolerance too tight**: Relax tolerance for difficult poses

### MPC Failures
Common issues:
1. **Infeasible constraints**: Check joint/velocity limits aren't too tight
2. **Poor warm start**: Clear `prev_solution` and retry
3. **Slow solve times**: Reduce horizon or relax IPOPT tolerances
4. **No convergence**: Increase Q/Q_terminal weights for tracking
5. **Oscillation**: Increase R weight for smoother motion

## Integration

### Before Running Demos
1. ✅ Run `test_forward_kinematics.py` - Verify FK is consistent
2. ✅ Run `test_inverse_kinematics.py` - Verify IK converges
3. ✅ Run `test_mpc_controller.py` - Verify MPC solves
4. ✅ Run `run_all_tests.py` - Comprehensive check

### FK in MPC
The current `FKSolver` uses MuJoCo internally, which:
- ✅ Guarantees perfect accuracy (0µm error)
- ❌ Cannot be used in symbolic CasADi optimization

Therefore, MPC uses:
- **Pure joint-space tracking** (Q, R, Q_terminal costs)
- **External collision detection** via MuJoCo
- **Waypoint guidance** for obstacle avoidance

To use symbolic FK in MPC, you would need to implement analytical DH-parameter-based FK that can be compiled into CasADi expressions.

## Continuous Integration

These tests should be run:
- Before committing changes to kinematics code
- After modifying robot model/URDF
- Before deploying to hardware
- As part of CI/CD pipeline

## Adding New Tests

To add a new test:
1. Create `test_<name>.py` in this directory
2. Implement test functions returning `bool` (pass/fail)
3. Add `if __name__ == "__main__"` block
4. Return exit code 0 for pass, 1 for fail
5. Run `run_all_tests.py` to include it automatically

## Notes

- Tests use random seeds for reproducibility
- MuJoCo FK is considered ground truth (< 0.1mm numerical error)
- All distances in meters, angles in radians
- Tests require MuJoCo model files in `sim/models/`

