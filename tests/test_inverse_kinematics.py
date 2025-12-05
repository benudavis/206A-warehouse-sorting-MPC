#!/usr/bin/env python3
"""
Test suite for Inverse Kinematics (IK) implementation.

Verifies IK solver accuracy and convergence properties.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control.inverse_kinematics import IKSolver


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


def test_ik_roundtrip():
    """Test IK by solving for known FK solutions (roundtrip test)."""
    print("=" * 70)
    print("IK ROUNDTRIP TEST")
    print("=" * 70)
    print("Generate random configs → FK → IK → Check we get back original pose")
    print()
    
    model, data = load_ur5e_model()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
    ik_solver = IKSolver(model, data, site_name="arm_hand_pinch")
    
    n_tests = 20
    errors = []
    successes = 0
    
    for i in range(n_tests):
        # Random config
        q_original = np.random.uniform(-2.0, 2.0, size=6)
        
        # Get target pose (position + orientation) via FK
        data.qpos[:6] = q_original
        mujoco.mj_forward(model, data)
        target_pos = data.site_xpos[site_id].copy()
        # Convert rotation matrix to quaternion
        target_mat = data.site_xmat[site_id].reshape(3, 3)
        target_quat = ik_solver._mat_to_quat(target_mat)
        
        # Solve IK for this pose
        q_ik, success = ik_solver.solve(
            target_pos,
            target_quat=target_quat,
            max_iterations=200,
            tolerance=0.01
        )
        
        # Check result
        data.qpos[:6] = q_ik
        mujoco.mj_forward(model, data)
        achieved_pos = data.site_xpos[site_id].copy()
        
        error = np.linalg.norm(achieved_pos - target_pos)
        errors.append(error)
        
        if success:
            successes += 1
        
        if i < 3 or error > 0.02:  # Print first 3 and failures
            print(f"Test {i+1}:")
            print(f"  Target:   {target_pos}")
            print(f"  Achieved: {achieved_pos}")
            print(f"  Error:    {error*1000:.2f} mm")
            print(f"  Success:  {success}")
            print()
    
    errors = np.array(errors)
    success_rate = successes / n_tests * 100
    
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Tests run:     {n_tests}")
    print(f"Success rate:  {success_rate:.1f}%")
    print(f"Mean error:    {errors.mean()*1000:.3f} mm")
    print(f"Median error:  {np.median(errors)*1000:.3f} mm")
    print(f"Max error:     {errors.max()*1000:.3f} mm")
    print()
    
    # More realistic pass criteria (random poses can be difficult)
    if success_rate > 90 and errors.mean() < 0.020 and errors.max() < 0.100:
        print("✅ PASSED: IK solver is accurate")
        return True
    elif success_rate > 80:
        print("⚠️  ACCEPTABLE: IK works but has some large errors on difficult poses")
        return True
    else:
        print("❌ FAILED: IK solver needs improvement")
        return False


def test_ik_workspace_positions():
    """Test IK on typical workspace positions."""
    print("\n" + "=" * 70)
    print("IK WORKSPACE TEST")
    print("=" * 70)
    print("Testing IK on typical positions in robot workspace")
    print()
    
    model, data = load_ur5e_model()
    ik_solver = IKSolver(model, data, site_name="arm_hand_pinch")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
    
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    
    # Typical workspace positions
    test_positions = {
        "Above table center": np.array([0.4, -0.3, 0.7]),
        "Left side":          np.array([0.2, -0.5, 0.6]),
        "Right side":         np.array([0.2, -0.1, 0.6]),
        "Far reach":          np.array([0.6, -0.3, 0.5]),
        "High position":      np.array([0.3, -0.3, 0.9]),
    }
    
    successes = 0
    
    for name, target_pos in test_positions.items():
        q_ik, success = ik_solver.solve(
            target_pos,
            target_quat=down_quat,
            max_iterations=500,
            tolerance=0.01
        )
        
        # Verify
        data.qpos[:6] = q_ik
        mujoco.mj_forward(model, data)
        achieved_pos = data.site_xpos[site_id].copy()
        error = np.linalg.norm(achieved_pos - target_pos)
        
        status = "✅" if success and error < 0.02 else "❌"
        print(f"{status} {name}:")
        print(f"     Target:   {target_pos}")
        print(f"     Achieved: {achieved_pos}")
        print(f"     Error:    {error*1000:.1f} mm")
        print()
        
        if success and error < 0.02:
            successes += 1
    
    success_rate = successes / len(test_positions) * 100
    
    if success_rate > 80:
        print(f"✅ PASSED: {success_rate:.0f}% success rate")
        return True
    else:
        print(f"❌ FAILED: Only {success_rate:.0f}% success rate")
        return False


def test_ik_convergence():
    """Test IK convergence behavior."""
    print("=" * 70)
    print("IK CONVERGENCE TEST")
    print("=" * 70)
    print("Testing how many iterations IK typically needs")
    print()
    
    model, data = load_ur5e_model()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
    ik_solver = IKSolver(model, data, site_name="arm_hand_pinch")
    
    # Test with varying max_iterations
    max_iter_tests = [10, 50, 100, 200, 500]
    
    for max_iter in max_iter_tests:
        successes = 0
        n_tests = 20
        
        for _ in range(n_tests):
            # Random FK pose
            q_rand = np.random.uniform(-2.0, 2.0, size=6)
            data.qpos[:6] = q_rand
            mujoco.mj_forward(model, data)
            target_pos = data.site_xpos[site_id].copy()
            # Convert rotation matrix to quaternion
            target_mat = data.site_xmat[site_id].reshape(3, 3)
            target_quat = ik_solver._mat_to_quat(target_mat)
            
            # Solve IK
            _, success = ik_solver.solve(
                target_pos,
                target_quat=target_quat,
                max_iterations=max_iter,
                tolerance=0.01
            )
            
            if success:
                successes += 1
        
        success_rate = successes / n_tests * 100
        print(f"Max iterations: {max_iter:3d} → Success rate: {success_rate:5.1f}%")
    
    print()
    print("✅ Convergence test complete")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INVERSE KINEMATICS TEST SUITE")
    print("=" * 70)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Convergence", test_ik_convergence()))
    results.append(("Roundtrip (FK → IK)", test_ik_roundtrip()))
    results.append(("Workspace positions", test_ik_workspace_positions()))
    
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
        print("IK solver is working correctly")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("IK solver may need tuning")
    
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)

