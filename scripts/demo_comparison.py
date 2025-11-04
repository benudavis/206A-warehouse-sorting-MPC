#!/usr/bin/env python3
"""
Visual demo comparing MPC vs learned controller
Shows side-by-side execution in simulation viewer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from src.control.mpc_controller import MPCController
from src.learning.mpc_imitator import MPCImitator
from src.perception.sim_state import SimulationState


def load_simulation():
    """Load MuJoCo simulation."""
    script_dir = Path(__file__).parent.parent / "sim"
    models_dir = script_dir / "models"
    
    scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))
    
    attachment_site = arm_spec.site('attachment_site')
    attachment_site.attach_body(hand_spec.worldbody, "hand_", "")

    robot_site = scene.site('robot_site')
    robot_site.attach_body(arm_spec.worldbody, "arm_", "")
    
    obj_spec = mujoco.MjSpec.from_file(str(models_dir / "cube.xml"))
    obj_frame = scene.worldbody.add_frame(pos=[0, -0.6, 0.5])
    obj_body = obj_frame.attach_body(obj_spec.worldbody, "obj_", "")
    obj_body.add_freejoint(name='obj_freejoint')

    model = scene.compile()
    model.opt.timestep = 0.0001
    data = mujoco.MjData(model)

    model.key_qpos[0][model.jnt('arm_shoulder_pan_joint').qposadr] += np.pi
    model.key_ctrl[0][model.jnt('arm_shoulder_pan_joint').dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    return model, data


def generate_random_target():
    """Generate random target joint configuration."""
    bounds = np.array([
        [-np.pi, np.pi],
        [-np.pi/2, np.pi/2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi]
    ])
    return np.random.uniform(bounds[:, 0], bounds[:, 1])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo MPC vs learned controller')
    parser.add_argument('--model', type=str, help='Path to trained model (.pth). If not provided, only MPC is shown')
    parser.add_argument('--mode', type=str, choices=['mpc', 'learned', 'both'], default='mpc',
                       help='Which controller to demonstrate')
    
    args = parser.parse_args()
    
    print("Loading simulation...")
    model, data = load_simulation()
    state_extractor = SimulationState(model, data)
    
    # Initialize controllers
    mpc_controller = None
    learned_controller = None
    
    if args.mode in ['mpc', 'both']:
        print("Initializing MPC controller...")
        mpc_controller = MPCController(n_joints=6, horizon=15, dt=0.01)
    
    if args.mode in ['learned', 'both'] and args.model:
        print("Loading learned controller...")
        learned_controller = MPCImitator()
        learned_controller.load(args.model)
    
    # Generate random target
    target_q = generate_random_target()
    print(f"\nTarget joint angles: {target_q}")
    
    ctrl_rate = 0.01
    rate = RateLimiter(frequency=1/ctrl_rate, warn=False)
    
    # Choose active controller
    if args.mode == 'learned' and learned_controller:
        active_controller = learned_controller
        controller_type = 'Learned'
    else:
        active_controller = mpc_controller
        controller_type = 'MPC'
    
    print(f"\nRunning with {controller_type} controller")
    print("Press Ctrl+C to stop")
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and step_count < 500:
            # Get state
            robot_state = state_extractor.get_robot_state()
            observation = state_extractor.get_observation_vector()
            
            # Compute action
            try:
                if controller_type == 'MPC':
                    action, pred_traj = active_controller.compute_control(robot_state, target_q)
                else:
                    action = active_controller.predict(observation)
                
                data.ctrl[:6] = action
            except Exception as e:
                print(f"Controller failed: {e}")
                break
            
            # Step simulation
            mujoco.mj_step(model, data, nstep=int(ctrl_rate / model.opt.timestep))
            viewer.sync()
            rate.sleep()
            
            # Check if reached target
            error = np.linalg.norm(data.qpos[:6] - target_q)
            
            if step_count % 50 == 0:
                print(f"Step {step_count}: error = {error:.4f} rad")
            
            if error < 0.05:
                print(f"\n✅ Target reached in {step_count} steps!")
                print("Generating new target...")
                target_q = generate_random_target()
                print(f"New target: {target_q}")
            
            step_count += 1


if __name__ == "__main__":
    main()

