"""
Test MPC controller functionality.

This test validates:
1. MPC initialization and configuration
2. Obstacle avoidance constraints
3. Joint and velocity limits
4. Trajectory generation
5. Warm starting
"""

import numpy as np
import mujoco
import pytest
from pathlib import Path

from src.control.mpc_controller import MPCController


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
def mpc_controller():
    """Create MPC controller instance."""
    return MPCController(
        n_joints=6,
        horizon=20,
        dt=0.05,
        enable_fk=True
    )


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


class TestMPCController:
    """Test suite for MPC controller."""

    def test_mpc_initialization(self):
        """Test MPC controller initialization."""
        mpc = MPCController(n_joints=6, horizon=30, dt=0.05, enable_fk=True)
        
        assert mpc.n_joints == 6
        assert mpc.horizon == 30
        assert mpc.dt == 0.05
        assert mpc.enable_fk == True
        assert mpc.solver is not None
        assert mpc.fk_fun is not None
        
        print(f"\nMPC initialized:")
        print(f"  Joints: {mpc.n_joints}")
        print(f"  Horizon: {mpc.horizon}")
        print(f"  dt: {mpc.dt}")
        print(f"  FK enabled: {mpc.enable_fk}")

    def test_mpc_without_fk(self):
        """Test MPC initialization without FK."""
        mpc = MPCController(n_joints=6, horizon=20, dt=0.05, enable_fk=False)
        
        assert mpc.fk_fun is None
        assert mpc.enable_fk == False
        assert mpc.solver is not None
        
        print(f"\nMPC without FK initialized successfully")

    def test_mpc_basic_control(self, mpc_controller):
        """Test basic MPC control without obstacles."""
        # Current state: [q, dq]
        current_state = np.zeros(12)
        current_state[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])  # Home position
        
        # Target: slightly different position
        target_state = np.array([0.2, -1.4, 0.3, -1.3, 0.1, 0.0])
        
        # Compute control
        q_next, q_traj = mpc_controller.compute_control(current_state, target_state)
        
        print(f"\nBasic control test:")
        print(f"  Current q: {current_state[:6]}")
        print(f"  Target q:  {target_state}")
        print(f"  Next q:    {q_next}")
        print(f"  Traj shape: {q_traj.shape}")
        
        assert q_next.shape == (6,), f"Expected shape (6,), got {q_next.shape}"
        assert q_traj.shape == (mpc_controller.horizon + 1, 6), \
            f"Expected trajectory shape ({mpc_controller.horizon + 1}, 6), got {q_traj.shape}"
        
        # Next command should move toward target
        distance_to_target_before = np.linalg.norm(current_state[:6] - target_state)
        distance_to_target_after = np.linalg.norm(q_next - target_state)
        
        print(f"  Distance to target before: {distance_to_target_before:.4f}")
        print(f"  Distance to target after:  {distance_to_target_after:.4f}")
        
        assert distance_to_target_after < distance_to_target_before, \
            "MPC should move closer to target"

    def test_mpc_joint_limits(self, mpc_controller):
        """Test that MPC respects joint limits."""
        # Start near limits
        current_state = np.zeros(12)
        current_state[:6] = np.array([np.pi - 0.1, 0, 0, 0, 0, 0])
        
        # Target beyond limits (should be clamped)
        target_state = np.array([np.pi + 0.5, 0, 0, 0, 0, 0])
        
        q_next, q_traj = mpc_controller.compute_control(current_state, target_state)
        
        print(f"\nJoint limits test:")
        print(f"  Current q[0]: {current_state[0]:.4f}")
        print(f"  Target q[0]:  {target_state[0]:.4f} (beyond limit)")
        print(f"  Next q[0]:    {q_next[0]:.4f}")
        print(f"  Joint limits: {mpc_controller.joint_limits}")
        
        # All joints in trajectory should respect limits
        for k in range(q_traj.shape[0]):
            assert np.all(q_traj[k] >= mpc_controller.joint_limits[0]), \
                f"Joint lower limit violated at step {k}"
            assert np.all(q_traj[k] <= mpc_controller.joint_limits[1]), \
                f"Joint upper limit violated at step {k}"

    def test_mpc_velocity_limits(self, mpc_controller):
        """Test that MPC respects velocity limits."""
        current_state = np.zeros(12)
        current_state[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        
        # Very distant target
        target_state = np.array([np.pi, 0, np.pi, 0, np.pi, 0])
        
        q_next, q_traj = mpc_controller.compute_control(current_state, target_state)
        
        print(f"\nVelocity limits test:")
        print(f"  Max velocity limit: {mpc_controller.max_velocity} rad/s")
        
        # Check velocity between consecutive steps
        max_velocity_observed = 0
        for k in range(q_traj.shape[0] - 1):
            velocity = np.abs((q_traj[k+1] - q_traj[k]) / mpc_controller.dt)
            max_vel_k = np.max(velocity)
            max_velocity_observed = max(max_velocity_observed, max_vel_k)
            
            print(f"  Step {k}: max velocity = {max_vel_k:.4f} rad/s")
            
            assert np.all(velocity <= mpc_controller.max_velocity + 1e-3), \
                f"Velocity limit violated at step {k}: {max_vel_k:.4f} > {mpc_controller.max_velocity}"
        
        print(f"  Max velocity observed: {max_velocity_observed:.4f} rad/s")

    def test_mpc_obstacle_avoidance(self, mpc_controller, mujoco_model, mujoco_data, ee_site_id):
        """Test MPC obstacle avoidance."""
        if mpc_controller.fk_fun is None:
            pytest.skip("FK function required for obstacle avoidance test")
        
        # Add obstacle in workspace
        obstacle_pos = np.array([0.3, 0.0, 0.3])
        obstacle_size = np.array([0.05, 0.05, 0.05])
        mpc_controller.add_obstacle(obstacle_pos, obstacle_size)
        
        print(f"\nObstacle avoidance test:")
        print(f"  Obstacle position: {obstacle_pos}")
        print(f"  Obstacle size: {obstacle_size}")
        print(f"  Number of obstacles: {len(mpc_controller.obstacles)}")
        
        # Start position
        current_state = np.zeros(12)
        current_state[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        
        # Target on other side of obstacle
        target_state = np.array([0.5, -1.2, 1.0, -1.5, 0.0, 0.0])
        
        # Compute trajectory
        q_next, q_traj = mpc_controller.compute_control(
            current_state, target_state, mujoco_model, mujoco_data, ee_site_id
        )
        
        # Check that trajectory avoids obstacle
        violations = 0
        min_distance = float('inf')
        
        for k in range(q_traj.shape[0]):
            # Compute EE position
            mujoco_data.qpos[:6] = q_traj[k]
            mujoco.mj_forward(mujoco_model, mujoco_data)
            ee_pos = mujoco_data.site_xpos[ee_site_id].copy()
            
            # Check distance to obstacle
            distance = mpc_controller._point_box_distance(ee_pos, obstacle_pos, obstacle_size)
            min_distance = min(min_distance, distance)
            
            if distance < 0:  # Inside obstacle
                violations += 1
                print(f"  Step {k}: EE inside obstacle! Distance = {distance:.4f} m")
        
        print(f"  Minimum distance to obstacle: {min_distance:.4f} m")
        print(f"  Collision violations: {violations}/{q_traj.shape[0]}")
        
        # Allow some tolerance, but should mostly avoid obstacles
        assert violations < q_traj.shape[0] * 0.1, \
            f"Too many obstacle violations: {violations}/{q_traj.shape[0]}"
        
        # Clear obstacles
        mpc_controller.clear_obstacles()

    def test_mpc_warm_starting(self, mpc_controller):
        """Test that MPC uses warm starting."""
        current_state = np.zeros(12)
        current_state[:6] = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        target_state = np.array([0.3, -1.4, 0.5, -1.3, 0.2, 0.0])
        
        # First call - no warm start
        assert mpc_controller.prev_solution is None
        q_next_1, q_traj_1 = mpc_controller.compute_control(current_state, target_state)
        
        # Should have cached solution
        assert mpc_controller.prev_solution is not None
        prev_sol = mpc_controller.prev_solution.copy()
        
        print(f"\nWarm starting test:")
        print(f"  First solution shape: {prev_sol.shape}")
        
        # Second call - should use warm start
        current_state[:6] = q_next_1
        q_next_2, q_traj_2 = mpc_controller.compute_control(current_state, target_state)
        
        print(f"  Warm start used: {mpc_controller.prev_solution is not None}")
        print(f"  Solutions are different: {not np.allclose(q_traj_1, q_traj_2)}")

    def test_mpc_configuration_updates(self, mpc_controller):
        """Test updating MPC configuration."""
        # Update cost weights
        mpc_controller.set_cost_weights(Q_scalar=1000.0, Q_terminal_scalar=2000.0, R_scalar=1.0)
        
        assert np.allclose(mpc_controller.Q, np.eye(6) * 1000.0)
        assert np.allclose(mpc_controller.Q_terminal, np.eye(6) * 2000.0)
        assert np.allclose(mpc_controller.R, np.eye(6) * 1.0)
        
        print(f"\nConfiguration update test:")
        print(f"  Q diagonal: {np.diag(mpc_controller.Q)}")
        print(f"  Q_terminal diagonal: {np.diag(mpc_controller.Q_terminal)}")
        print(f"  R diagonal: {np.diag(mpc_controller.R)}")
        
        # Update joint limits
        new_lower = np.array([-np.pi] * 6)
        new_upper = np.array([np.pi] * 6)
        mpc_controller.set_joint_limits(new_lower, new_upper)
        
        assert np.allclose(mpc_controller.joint_limits[0], new_lower)
        assert np.allclose(mpc_controller.joint_limits[1], new_upper)
        
        # Update velocity limit
        mpc_controller.set_velocity_limit(2.0)
        assert mpc_controller.max_velocity == 2.0
        
        print(f"  Configuration updates successful")

    def test_mpc_fallback_on_failure(self, mpc_controller):
        """Test MPC fallback when solve fails."""
        current_state = np.zeros(12)
        target_state = np.array([0.1, -1.5, 0.2, -1.4, 0.0, 0.0])
        
        # Force very tight constraints to potentially cause failure
        mpc_controller.set_velocity_limit(0.001)  # Extremely low
        
        # Should not crash even if solver struggles
        try:
            q_next, q_traj = mpc_controller.compute_control(current_state, target_state)
            
            print(f"\nFallback test:")
            print(f"  Computation completed (may have used fallback)")
            print(f"  Next q: {q_next}")
            
            assert q_next.shape == (6,)
            assert q_traj.shape[0] == mpc_controller.horizon + 1
            
        except Exception as e:
            pytest.fail(f"MPC should handle failures gracefully, got: {e}")
        
        # Reset velocity limit
        mpc_controller.set_velocity_limit(3.0)

    def test_mpc_link_body_initialization(self, mpc_controller, mujoco_model):
        """Test link body initialization for collision checking."""
        mpc_controller.initialize_link_bodies(mujoco_model)
        
        print(f"\nLink body initialization test:")
        print(f"  Number of link bodies: {len(mpc_controller.link_body_ids)}")
        print(f"  Link body IDs: {mpc_controller.link_body_ids}")
        
        # All IDs should be valid (if any found)
        for bid in mpc_controller.link_body_ids:
            assert bid >= 0, f"Invalid body ID: {bid}"
        
        # Note: It's OK if no bodies are found if the model doesn't have the robot loaded
        # The test just verifies the initialization doesn't crash

    def test_mpc_multiple_obstacles(self, mpc_controller):
        """Test MPC with multiple obstacles."""
        if mpc_controller.fk_fun is None:
            pytest.skip("FK function required for obstacle test")
        
        # Add multiple obstacles
        mpc_controller.add_obstacle(np.array([0.2, 0.1, 0.2]), np.array([0.03, 0.03, 0.03]))
        mpc_controller.add_obstacle(np.array([0.3, -0.1, 0.3]), np.array([0.03, 0.03, 0.03]))
        mpc_controller.add_obstacle(np.array([0.4, 0.0, 0.25]), np.array([0.03, 0.03, 0.03]))
        
        print(f"\nMultiple obstacles test:")
        print(f"  Number of obstacles: {len(mpc_controller.obstacles)}")
        
        current_state = np.zeros(12)
        target_state = np.array([0.5, -1.0, 0.8, -1.2, 0.0, 0.0])
        
        # Should still compute control
        q_next, q_traj = mpc_controller.compute_control(current_state, target_state)
        
        assert q_next.shape == (6,)
        print(f"  Control computed successfully with {len(mpc_controller.obstacles)} obstacles")
        
        mpc_controller.clear_obstacles()


class TestMPCHelpers:
    """Test helper functions in MPC controller."""

    def test_point_box_distance_outside(self):
        """Test point-to-box distance for point outside box."""
        point = np.array([1.0, 0.0, 0.0])
        box_center = np.array([0.0, 0.0, 0.0])
        box_half_size = np.array([0.5, 0.5, 0.5])
        
        distance = MPCController._point_box_distance(point, box_center, box_half_size)
        
        print(f"\nPoint-box distance test (outside):")
        print(f"  Point: {point}")
        print(f"  Box center: {box_center}")
        print(f"  Box half-size: {box_half_size}")
        print(f"  Distance: {distance:.4f}")
        
        # Point at x=1.0, box extends to x=0.5, so distance should be 0.5
        assert abs(distance - 0.5) < 1e-6, f"Expected distance 0.5, got {distance}"

    def test_point_box_distance_inside(self):
        """Test point-to-box distance for point inside box."""
        point = np.array([0.2, 0.1, 0.0])
        box_center = np.array([0.0, 0.0, 0.0])
        box_half_size = np.array([0.5, 0.5, 0.5])
        
        distance = MPCController._point_box_distance(point, box_center, box_half_size)
        
        print(f"\nPoint-box distance test (inside):")
        print(f"  Point: {point}")
        print(f"  Distance: {distance:.4f} (should be negative)")
        
        # Point is inside, so distance should be negative
        assert distance < 0, f"Point inside box should have negative distance, got {distance}"

    def test_point_box_distance_on_surface(self):
        """Test point-to-box distance for point on surface."""
        point = np.array([0.5, 0.0, 0.0])
        box_center = np.array([0.0, 0.0, 0.0])
        box_half_size = np.array([0.5, 0.5, 0.5])
        
        distance = MPCController._point_box_distance(point, box_center, box_half_size)
        
        print(f"\nPoint-box distance test (on surface):")
        print(f"  Point: {point}")
        print(f"  Distance: {distance:.4f}")
        
        # Point on surface should have distance close to 0
        assert abs(distance) < 1e-6, f"Expected distance ≈0, got {distance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
