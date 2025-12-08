"""
Test forward kinematics accuracy against MuJoCo.

This test compares the analytic FK implementation with MuJoCo's FK
to ensure they produce consistent end-effector positions.
"""

import numpy as np
import mujoco
import pytest
from pathlib import Path

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
        return site_id
    except Exception:
        pytest.skip("End-effector site 'arm_hand_pinch' not found in model")


class TestForwardKinematics:
    """Test suite for forward kinematics."""

    def test_fk_zero_configuration(self, fk_function, mujoco_model, mujoco_data, ee_site_id):
        """Test FK at zero configuration."""
        if mujoco_model.nq < 6:
            pytest.skip("Model has fewer than 6 DOFs (robot not loaded)")
        
        q_test = np.zeros(6)
        
        # Analytic FK
        fk_pos = np.array(fk_function(q_test)).flatten()
        
        # MuJoCo FK
        mujoco_data.qpos[:6] = q_test
        mujoco.mj_forward(mujoco_model, mujoco_data)
        mujoco_pos = mujoco_data.site_xpos[ee_site_id].copy()
        
        # Compare
        error = np.linalg.norm(fk_pos - mujoco_pos)
        print(f"\nZero config - Analytic FK: {fk_pos}")
        print(f"Zero config - MuJoCo FK:   {mujoco_pos}")
        print(f"Zero config - Error: {error:.6f} m")
        
        # FK calibrated to match at zero config (should be near-perfect)
        assert error < 0.005, f"FK error at zero config: {error:.6f} m"

    def test_fk_random_configurations(self, fk_function, mujoco_model, mujoco_data, ee_site_id):
        """Test FK at random configurations."""
        if mujoco_model.nq < 6:
            pytest.skip("Model has fewer than 6 DOFs (robot not loaded)")
        
        np.random.seed(42)
        n_tests = 100
        errors = []
        
        for i in range(n_tests):
            # Random joint configuration
            q_test = np.random.uniform(-np.pi, np.pi, 6)
            
            # Analytic FK
            fk_pos = np.array(fk_function(q_test)).flatten()
            
            # MuJoCo FK
            mujoco_data.qpos[:6] = q_test
            mujoco.mj_forward(mujoco_model, mujoco_data)
            mujoco_pos = mujoco_data.site_xpos[ee_site_id].copy()
            
            # Compute error
            error = np.linalg.norm(fk_pos - mujoco_pos)
            errors.append(error)
            
            if i < 5:  # Print first 5
                print(f"\nTest {i+1} - q: {q_test}")
                print(f"Test {i+1} - Analytic FK: {fk_pos}")
                print(f"Test {i+1} - MuJoCo FK:   {mujoco_pos}")
                print(f"Test {i+1} - Error: {error:.6f} m")
        
        errors = np.array(errors)
        print(f"\n=== Random Configuration Tests (n={n_tests}) ===")
        print(f"Mean error:   {np.mean(errors):.6f} m")
        print(f"Median error: {np.median(errors):.6f} m")
        print(f"Max error:    {np.max(errors):.6f} m")
        print(f"Min error:    {np.min(errors):.6f} m")
        print(f"Std dev:      {np.std(errors):.6f} m")
        
        # FK should match MuJoCo at the pinch point to within a few millimeters
        assert np.mean(errors) < 0.005, f"Mean FK error too large: {np.mean(errors):.6f} m"
        assert np.max(errors) < 0.010, f"Max FK error too large: {np.max(errors):.6f} m"
        assert np.std(errors) < 0.003, f"FK errors too variable: {np.std(errors):.6f} m"

    def test_fk_workspace_corners(self, fk_function, mujoco_model, mujoco_data, ee_site_id):
        """Test FK at typical workspace positions."""
        if mujoco_model.nq < 6:
            pytest.skip("Model has fewer than 6 DOFs (robot not loaded)")
        
        # Common UR5e configurations
        test_configs = [
            np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0]),  # Home position
            np.array([np.pi/4, -np.pi/3, np.pi/4, -np.pi/2, 0, 0]),  # Front right
            np.array([-np.pi/4, -np.pi/3, np.pi/4, -np.pi/2, 0, 0]),  # Front left
            np.array([np.pi, -np.pi/4, -np.pi/4, -3*np.pi/4, 0, 0]),  # Back
            np.array([0, -np.pi/6, np.pi/3, -np.pi/2, np.pi/2, 0]),  # Overhead
        ]
        
        config_names = ["Home", "Front Right", "Front Left", "Back", "Overhead"]
        
        for i, (q_test, name) in enumerate(zip(test_configs, config_names)):
            # Analytic FK
            fk_pos = np.array(fk_function(q_test)).flatten()
            
            # MuJoCo FK
            mujoco_data.qpos[:6] = q_test
            mujoco.mj_forward(mujoco_model, mujoco_data)
            mujoco_pos = mujoco_data.site_xpos[ee_site_id].copy()
            
            # Compute error
            error = np.linalg.norm(fk_pos - mujoco_pos)
            
            print(f"\n{name} - Analytic FK: {fk_pos}")
            print(f"{name} - MuJoCo FK:   {mujoco_pos}")
            print(f"{name} - Error: {error:.6f} m")
            
            # FK should be accurate at typical workspace configurations (few mm)
            assert error < 0.010, f"FK error at {name}: {error:.6f} m"

    def test_fk_joint_limits(self, fk_function, mujoco_model, mujoco_data, ee_site_id):
        """Test FK at joint limits."""
        if mujoco_model.nq < 6:
            pytest.skip("Model has fewer than 6 DOFs (robot not loaded)")
        
        # Test near limits (avoiding singularities)
        limit_configs = [
            np.array([np.pi, 0, 0, 0, 0, 0]),
            np.array([-np.pi, 0, 0, 0, 0, 0]),
            np.array([0, -np.pi, 0, 0, 0, 0]),
            np.array([0, 0, np.pi, 0, 0, 0]),
        ]
        
        for i, q_test in enumerate(limit_configs):
            # Analytic FK
            fk_pos = np.array(fk_function(q_test)).flatten()
            
            # MuJoCo FK
            mujoco_data.qpos[:6] = q_test
            mujoco.mj_forward(mujoco_model, mujoco_data)
            mujoco_pos = mujoco_data.site_xpos[ee_site_id].copy()
            
            # Compute error
            error = np.linalg.norm(fk_pos - mujoco_pos)
            
            print(f"\nLimit config {i+1} - q: {q_test}")
            print(f"Limit config {i+1} - Analytic FK: {fk_pos}")
            print(f"Limit config {i+1} - MuJoCo FK:   {mujoco_pos}")
            print(f"Limit config {i+1} - Error: {error:.6f} m")
            
            assert error < 0.010, f"FK error at limit config {i+1}: {error:.6f} m"

    def test_fk_output_shape(self, fk_function):
        """Test that FK returns correct output shape."""
        q_test = np.zeros(6)
        result = np.array(fk_function(q_test))
        
        assert result.shape == (3, 1) or result.shape == (3,), \
            f"FK should return 3D position, got shape {result.shape}"

    def test_fk_casadi_symbolic(self):
        """Test that FK function is CasADi symbolic."""
        import casadi as ca
        
        fk_fun = build_ur5e_fk_function()
        
        # Create symbolic input
        q_sym = ca.SX.sym("q", 6)
        result = fk_fun(q_sym)
        
        # Should be symbolic expression
        assert isinstance(result, ca.SX), "FK should work with symbolic CasADi variables"
        
        # Should be able to take jacobian (for MPC gradients)
        jac = ca.jacobian(result, q_sym)
        assert jac.shape == (3, 6), f"Jacobian should be 3x6, got {jac.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
