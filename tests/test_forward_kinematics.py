#!/usr/bin/env python3
"""
Test suite for Forward Kinematics (FK) implementation.

Verifies that analytical FK matches MuJoCo FK to within acceptable tolerances.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control.forward_kinematics import FKSolver


def load_ur5e_model():
    """Load UR5e + gripper model."""
    MODELS_DIR = Path(__file__).parent.parent / "sim" / "models"
    scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robotiq_2f85" / "2f85.xml"))
    
    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
    
    model = scene.compile()
    data = mujoco.MjData(model)
    
    return model, data


def test_fk_accuracy():
    """Test FK consistency on random configurations."""
    print("=" * 70)
    print("FK CONSISTENCY TEST")
    print("=" * 70)
    print("NOTE: FKSolver uses MuJoCo internally, so errors should be ~0")
    print()
    
    model, data = load_ur5e_model()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
    fk_solver = FKSolver(model=model, site_name="arm_hand_pinch")
    
    n_tests = 100
    errors = []
    max_error = 0.0
    max_error_q = None
    
    print(f"Testing {n_tests} random joint configurations...")
    print()
    
    for i in range(n_tests):
        # Random configuration
        q = np.random.uniform(-2.0, 2.0, size=6)
        
        # MuJoCo FK (direct)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        ee_mujoco = data.site_xpos[site_id].copy()
        
        # FK Solver (should be identical)
        ee_fk = fk_solver.compute_ee_position(q)
        
        # Error (should be numerical noise only)
        error = np.linalg.norm(ee_fk - ee_mujoco)
        errors.append(error)
        
        if error > max_error:
            max_error = error
            max_error_q = q.copy()
        
        # Print first 3 and any anomalies
        if i < 3 or error > 1e-6:
            print(f"Test {i+1}:")
            print(f"  q = {q}")
            print(f"  MuJoCo:  {ee_mujoco}")
            print(f"  FKSolver: {ee_fk}")
            print(f"  Error: {error*1e6:.2f} µm (micrometers)")
            print()
    
    errors = np.array(errors)
    
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Tests run:    {n_tests}")
    print(f"Mean error:   {errors.mean()*1e6:.3f} µm")
    print(f"Median error: {np.median(errors)*1e6:.3f} µm")
    print(f"Max error:    {max_error*1e6:.3f} µm")
    print()
    
    # Since FKSolver uses MuJoCo, errors should be floating-point noise only
    if max_error > 1e-3:  # 1mm
        print("❌ FAILED: FK has unexpected discrepancy")
        print(f"   Check FKSolver implementation!")
        return False
    else:
        print("✅ PASSED: FK is consistent with MuJoCo")
        print(f"   (as expected, since it uses MuJoCo internally)")
        return True


def test_fk_multi_link():
    """Test FK for all links (shoulder, elbow, wrist, EE)."""
    print("\n" + "=" * 70)
    print("MULTI-LINK FK TEST")
    print("=" * 70)
    
    model, data = load_ur5e_model()
    fk_solver = FKSolver(model=model, site_name="arm_hand_pinch")
    
    # Test configuration
    q = np.array([0.5, -1.0, 1.5, -0.5, -1.5, 0.0])
    
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    
    # Get MuJoCo positions for link bodies (must match LinkNames defaults in FKSolver)
    link_names = {
        'shoulder': 'arm_shoulder_link',
        'elbow': 'arm_forearm_link',
        'wrist': 'arm_wrist_3_link',  # Default in FKSolver
    }
    
    print(f"Joint config: {q}")
    print()
    
    all_good = True
    for link_key, body_name in link_names.items():
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos_mujoco = data.xpos[body_id].copy()
            
            # Get analytical FK
            if link_key == 'shoulder':
                pos_analytical = fk_solver.compute_shoulder_position(q)
            elif link_key == 'elbow':
                pos_analytical = fk_solver.compute_elbow_position(q)
            elif link_key == 'wrist':
                pos_analytical = fk_solver.compute_wrist_position(q)
            
            error = np.linalg.norm(pos_analytical - pos_mujoco)
            
            print(f"{link_key}:")
            print(f"  MuJoCo:    {pos_mujoco}")
            print(f"  FKSolver:  {pos_analytical}")
            print(f"  Error: {error*1e6:.2f} µm")
            
            if error > 1e-3:  # 1mm
                print(f"  ❌ Unexpected error!")
                all_good = False
            else:
                print(f"  ✅ Good (< 1mm)")
            print()
            
        except Exception as e:
            print(f"{link_key}: ⚠️  Could not test ({e})")
            all_good = False
    
    return all_good


def test_fk_home_position():
    """Test FK at home position (all joints = 0)."""
    print("=" * 70)
    print("FK HOME POSITION TEST")
    print("=" * 70)
    
    model, data = load_ur5e_model()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
    fk_solver = FKSolver(model=model, site_name="arm_hand_pinch")
    
    q_home = np.zeros(6)
    
    # MuJoCo FK
    data.qpos[:6] = q_home
    mujoco.mj_forward(model, data)
    ee_mujoco = data.site_xpos[site_id].copy()
    
    # FKSolver
    ee_fk = fk_solver.compute_ee_position(q_home)
    
    error = np.linalg.norm(ee_fk - ee_mujoco)
    
    print(f"Home position (all joints = 0):")
    print(f"  MuJoCo:  {ee_mujoco}")
    print(f"  FKSolver: {ee_fk}")
    print(f"  Error: {error*1e6:.3f} µm")
    print()
    
    if error < 1e-6:
        print("✅ PASSED: FK is identical to MuJoCo (as expected)")
        return True
    else:
        print("⚠️  Warning: Small numerical difference detected")
        return error < 1e-3  # Still pass if < 1mm


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FORWARD KINEMATICS TEST SUITE")
    print("=" * 70)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Home Position", test_fk_home_position()))
    results.append(("Consistency (100 random configs)", test_fk_accuracy()))
    results.append(("Multi-link FK", test_fk_multi_link()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("FK implementation is ready for MPC")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("FK implementation needs fixes before using in MPC")
    
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)

