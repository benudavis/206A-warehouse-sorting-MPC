"""
Test inverse kinematics accuracy.

This test validates the IK solver by:
1. Testing IK convergence for reachable targets
2. Verifying FK(IK(target)) ≈ target
3. Testing orientation IK
"""

import numpy as np
import mujoco
import pytest
from pathlib import Path

from src.control.inverse_kinematics import IKSolver
from src.control.forward_kinematics import build_ur5e_fk_function


@pytest.fixture
def mujoco_model():
    """Build MuJoCo model with robot for testing (same as demos)."""
    models_dir = Path(__file__).parent.parent / "sim" / "models"
    scene_path = models_dir / "scene.xml"
    arm_path = models_dir / "universal_robots_ur5e" / "ur5e.xml"
    hand_path = models_dir / "robotiq_2f85" / "2f85.xml"
    
    if not scene_path.exists():
        pytest.skip(f"Scene model not found at {scene_path}")
    if not arm_path.exists():
        pytest.skip(f"Arm model not found at {arm_path}")
    if not hand_path.exists():
        pytest.skip(f"Hand model not found at {hand_path}")
    
    # Build world with robot (same as demo scripts)
    scene = mujoco.MjSpec.from_file(str(scene_path))
    arm_spec = mujoco.MjSpec.from_file(str(arm_path))
    hand_spec = mujoco.MjSpec.from_file(str(hand_path))
    
    # Attach hand to arm, arm to scene
    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
    
    return scene.compile()


@pytest.fixture
def mujoco_data(mujoco_model):
    """Create MuJoCo data instance."""
    return mujoco.MjData(mujoco_model)


@pytest.fixture
def ik_solver(mujoco_model, mujoco_data):
    """Create IK solver instance."""
    return IKSolver(mujoco_model, mujoco_data, site_name="arm_hand_pinch")


@pytest.fixture
def fk_function():
    """Build CasADi FK function."""
    return build_ur5e_fk_function()


@pytest.fixture
def ee_site_id(mujoco_model):
    """Get end-effector site ID."""
    try:
        site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
        if site_id < 0:
            pytest.skip("End-effector site 'arm_hand_pinch' not found in model")
        if mujoco_model.nq < 6:
            pytest.skip("Model has fewer than 6 DOFs (robot not loaded)")
        return site_id
    except Exception:
        pytest.skip("End-effector site 'arm_hand_pinch' not found in model")


class TestInverseKinematics:
    """Test suite for inverse kinematics."""

    def test_ik_identity(self, ik_solver, mujoco_model, mujoco_data, ee_site_id):
        """Test that IK can recover current position."""
        # Set random initial configuration
        q_initial = np.array([0.5, -0.8, 1.2, -1.5, 0.3, 0.0])
        mujoco_data.qpos[:6] = q_initial
        mujoco.mj_forward(mujoco_model, mujoco_data)
        
        # Get current EE position
        target_pos = mujoco_data.site_xpos[ee_site_id].copy()
        
        # Solve IK for same position
        q_solved, success = ik_solver.solve(target_pos, max_iterations=50)
        
        print(f"\nInitial q: {q_initial}")
        print(f"Solved q:  {q_solved}")
        print(f"Success:   {success}")
        
        # Check that solution reaches target
        mujoco_data.qpos[:6] = q_solved
        mujoco.mj_forward(mujoco_model, mujoco_data)
        achieved_pos = mujoco_data.site_xpos[ee_site_id].copy()
        
        error = np.linalg.norm(achieved_pos - target_pos)
        print(f"Target pos:   {target_pos}")
        print(f"Achieved pos: {achieved_pos}")
        print(f"Error: {error:.6f} m")
        
        assert success, "IK should succeed for current position"
        assert error < 0.02, f"IK position error too large: {error:.6f} m"

    def test_ik_random_reachable_targets(self, ik_solver, mujoco_model, mujoco_data, ee_site_id):
        """Test IK for random reachable targets."""
        np.random.seed(42)
        n_tests = 20
        successes = 0
        errors = []
        
        for i in range(n_tests):
            # Generate random reachable target by random joint config
            q_random = np.random.uniform(-np.pi, np.pi, 6)
            mujoco_data.qpos[:6] = q_random
            mujoco.mj_forward(mujoco_model, mujoco_data)
            target_pos = mujoco_data.site_xpos[ee_site_id].copy()
            
            # Reset to home position before IK
            mujoco_data.qpos[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
            
            # Solve IK
            q_solved, success = ik_solver.solve(target_pos, max_iterations=100)
            
            if success:
                successes += 1
                
                # Verify solution
                mujoco_data.qpos[:6] = q_solved
                mujoco.mj_forward(mujoco_model, mujoco_data)
                achieved_pos = mujoco_data.site_xpos[ee_site_id].copy()
                
                error = np.linalg.norm(achieved_pos - target_pos)
                errors.append(error)
                
                if i < 3:  # Print first 3
                    print(f"\nTest {i+1}:")
                    print(f"  Target pos:   {target_pos}")
                    print(f"  Achieved pos: {achieved_pos}")
                    print(f"  Error: {error:.6f} m")
                    print(f"  Success: {success}")
        
        success_rate = successes / n_tests
        print(f"\n=== Random Reachable Targets (n={n_tests}) ===")
        print(f"Success rate: {success_rate:.1%} ({successes}/{n_tests})")
        
        if len(errors) > 0:
            errors = np.array(errors)
            print(f"Mean error:   {np.mean(errors):.6f} m")
            print(f"Median error: {np.median(errors):.6f} m")
            print(f"Max error:    {np.max(errors):.6f} m")
            print(f"Min error:    {np.min(errors):.6f} m")
        
        assert success_rate >= 0.8, f"IK success rate too low: {success_rate:.1%}"
        if len(errors) > 0:
            assert np.mean(errors) < 0.02, f"Mean IK error too large: {np.mean(errors):.6f} m"

    def test_ik_with_fk_roundtrip(self, ik_solver, fk_function, mujoco_model, mujoco_data, ee_site_id):
        """Test IK → FK roundtrip: FK(IK(target)) ≈ target."""
        np.random.seed(123)
        n_tests = 15
        errors_mujoco = []
        errors_analytic = []
        
        for i in range(n_tests):
            # Random target position (reachable)
            q_random = np.random.uniform(-np.pi, np.pi, 6)
            mujoco_data.qpos[:6] = q_random
            mujoco.mj_forward(mujoco_model, mujoco_data)
            target_pos = mujoco_data.site_xpos[ee_site_id].copy()
            
            # Reset to home
            mujoco_data.qpos[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
            
            # Solve IK
            q_solved, success = ik_solver.solve(target_pos, max_iterations=100)
            
            if not success:
                continue
            
            # Check with MuJoCo FK
            mujoco_data.qpos[:6] = q_solved
            mujoco.mj_forward(mujoco_model, mujoco_data)
            achieved_mujoco = mujoco_data.site_xpos[ee_site_id].copy()
            error_mujoco = np.linalg.norm(achieved_mujoco - target_pos)
            errors_mujoco.append(error_mujoco)
            
            # Check with analytic FK
            achieved_analytic = np.array(fk_function(q_solved)).flatten()
            error_analytic = np.linalg.norm(achieved_analytic - target_pos)
            errors_analytic.append(error_analytic)
            
            if i < 3:
                print(f"\nRoundtrip test {i+1}:")
                print(f"  Target:         {target_pos}")
                print(f"  MuJoCo FK:      {achieved_mujoco}")
                print(f"  Analytic FK:    {achieved_analytic}")
                print(f"  Error (MuJoCo): {error_mujoco:.6f} m")
                print(f"  Error (Analytic): {error_analytic:.6f} m")
        
        errors_mujoco = np.array(errors_mujoco)
        errors_analytic = np.array(errors_analytic)
        
        print(f"\n=== IK → FK Roundtrip Tests (n={len(errors_mujoco)}) ===")
        print(f"MuJoCo FK errors:   mean={np.mean(errors_mujoco):.6f} m, max={np.max(errors_mujoco):.6f} m")
        print(f"Analytic FK errors: mean={np.mean(errors_analytic):.6f} m, max={np.max(errors_analytic):.6f} m")
        
        assert len(errors_mujoco) >= n_tests * 0.8, f"Too few IK solutions: {len(errors_mujoco)}/{n_tests}"
        # Jacobian IK typically achieves 5-10mm accuracy, which is excellent
        assert np.mean(errors_mujoco) < 0.010, "IK → MuJoCo FK roundtrip error too large"
        assert np.max(errors_mujoco) < 0.015, "IK → MuJoCo FK max roundtrip error too large"
        
        assert np.mean(errors_analytic) < 0.010, "IK → Analytic FK roundtrip error too large"
        assert np.max(errors_analytic) < 0.015, "IK → Analytic FK max roundtrip error too large"

    def test_ik_specific_positions(self, ik_solver, mujoco_model, mujoco_data, ee_site_id):
        """Test IK for specific known positions."""
        test_positions = [
            np.array([0.4, 0.0, 0.3]),   # Front center
            np.array([0.3, 0.3, 0.2]),   # Front right
            np.array([0.3, -0.3, 0.2]),  # Front left
            np.array([0.2, 0.0, 0.5]),   # High center
        ]
        
        position_names = ["Front Center", "Front Right", "Front Left", "High Center"]
        
        for pos, name in zip(test_positions, position_names):
            # Reset to home
            mujoco_data.qpos[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
            
            # Solve IK
            q_solved, success = ik_solver.solve(pos, max_iterations=100)
            
            if success:
                # Verify
                mujoco_data.qpos[:6] = q_solved
                mujoco.mj_forward(mujoco_model, mujoco_data)
                achieved = mujoco_data.site_xpos[ee_site_id].copy()
                error = np.linalg.norm(achieved - pos)
                
                print(f"\n{name}:")
                print(f"  Target:   {pos}")
                print(f"  Achieved: {achieved}")
                print(f"  Error:    {error:.6f} m")
                print(f"  q:        {q_solved}")
                
                assert error < 0.02, f"IK error at {name}: {error:.6f} m"
            else:
                print(f"\n{name}: IK did not converge (position may be unreachable)")

    def test_ik_with_orientation(self, ik_solver, mujoco_model, mujoco_data, ee_site_id):
        """Test IK with orientation constraint."""
        # Get a reachable pose
        q_test = np.array([0.5, -1.0, 1.5, -1.5, 0.0, 0.0])
        mujoco_data.qpos[:6] = q_test
        mujoco.mj_forward(mujoco_model, mujoco_data)
        
        target_pos = mujoco_data.site_xpos[ee_site_id].copy()
        target_mat = mujoco_data.site_xmat[ee_site_id].reshape(3, 3)
        target_quat = ik_solver._mat_to_quat(target_mat)
        
        # Reset
        mujoco_data.qpos[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        
        # Solve IK with orientation
        q_solved, success = ik_solver.solve(
            target_pos, 
            target_quat=target_quat,
            max_iterations=150,
            tolerance=0.01
        )
        
        print(f"\nOrientation IK test:")
        print(f"  Target q:     {q_test}")
        print(f"  Solved q:     {q_solved}")
        print(f"  Success:      {success}")
        
        if success:
            mujoco_data.qpos[:6] = q_solved
            mujoco.mj_forward(mujoco_model, mujoco_data)
            achieved_pos = mujoco_data.site_xpos[ee_site_id].copy()
            achieved_mat = mujoco_data.site_xmat[ee_site_id].reshape(3, 3)
            
            pos_error = np.linalg.norm(achieved_pos - target_pos)
            ori_error = np.linalg.norm(achieved_mat - target_mat, 'fro')
            
            print(f"  Pos error:    {pos_error:.6f} m")
            print(f"  Ori error:    {ori_error:.6f} (Frobenius norm)")
            
            assert pos_error < 0.02, f"Position error with orientation IK: {pos_error:.6f} m"
            assert ori_error < 0.1, f"Orientation error: {ori_error:.6f}"

    def test_ik_solver_initialization(self, mujoco_model, mujoco_data):
        """Test IK solver initialization."""
        solver = IKSolver(mujoco_model, mujoco_data, site_name="arm_hand_pinch")
        
        assert solver.model is not None
        assert solver.data is not None
        assert solver.site_id >= 0
        assert solver.site_name == "arm_hand_pinch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
