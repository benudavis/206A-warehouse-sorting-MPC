#!/usr/bin/env python3
"""
Collect MPC training data from simulation
Runs MPC controller and records (observation, action) pairs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
from loop_rate_limiters import RateLimiter
import json
from datetime import datetime

from src.control.mpc_controller import MPCController
from src.perception.sim_state import SimulationState


def load_simulation():
    """Load MuJoCo simulation."""
    script_dir = Path(__file__).parent.parent / "sim"
    models_dir = script_dir / "models"
    
    # Load models
    scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))
    
    # Attach hand
    attachment_site = arm_spec.site('attachment_site')
    attachment_site.attach_body(hand_spec.worldbody, "hand_", "")

    # Merge arm and hand into scene
    robot_site = scene.site('robot_site')
    robot_site.attach_body(arm_spec.worldbody, "arm_", "")
    
    # Add cube object
    obj_spec = mujoco.MjSpec.from_file(str(models_dir / "cube.xml"))
    obj_frame = scene.worldbody.add_frame(pos=[0, -0.6, 0.5])
    obj_body = obj_frame.attach_body(obj_spec.worldbody, "obj_", "")
    obj_body.add_freejoint(name='obj_freejoint')

    model = scene.compile()
    model.opt.timestep = 0.0001
    data = mujoco.MjData(model)

    # Reset to home position
    model.key_qpos[0][model.jnt('arm_shoulder_pan_joint').qposadr] += np.pi
    model.key_ctrl[0][model.jnt('arm_shoulder_pan_joint').dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    return model, data


def generate_random_target(bounds=None):
    """Generate random target joint configuration."""
    if bounds is None:
        bounds = np.array([
            [-np.pi, np.pi],
            [-np.pi/2, np.pi/2],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ])
    
    target = np.random.uniform(bounds[:, 0], bounds[:, 1])
    return target


def collect_data(n_episodes=100, steps_per_episode=200, render=False):
    """
    Collect MPC data by running controller in simulation.
    
    Args:
        n_episodes: Number of episodes to collect
        steps_per_episode: Steps per episode
        render: Whether to render simulation
        
    Returns:
        observations: Array of observations (N, obs_dim)
        actions: Array of MPC actions (N, action_dim)
        metadata: Collection metadata
    """
    print("Loading simulation...")
    model, data = load_simulation()
    state_extractor = SimulationState(model, data)
    
    print("Initializing MPC controller...")
    mpc = MPCController(n_joints=6, horizon=15, dt=0.01)
    
    # Storage
    observations = []
    actions = []
    targets = []
    
    ctrl_rate = 0.01
    rate = RateLimiter(frequency=1/ctrl_rate, warn=False)
    
    print(f"Collecting {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        # Reset to random initial position
        initial_q = generate_random_target()
        data.qpos[:6] = initial_q
        data.qvel[:6] = 0
        mujoco.mj_forward(model, data)
        
        # Random target
        target_q = generate_random_target()
        
        for step in range(steps_per_episode):
            # Get current state
            robot_state = state_extractor.get_robot_state()
            observation = state_extractor.get_observation_vector(target_q=target_q)  # Include target!
            
            # Compute MPC action
            try:
                action, pred_traj = mpc.compute_control(robot_state, target_q)
                
                # Record data
                observations.append(observation)
                actions.append(action)
                targets.append(target_q)
                
                # Apply control
                data.ctrl[:6] = action
                
            except Exception as e:
                print(f"MPC failed at episode {episode}, step {step}: {e}")
                break
            
            # Step simulation
            mujoco.mj_step(model, data, nstep=int(ctrl_rate / model.opt.timestep))
            
            if render:
                # TODO: Add viewer if needed
                pass
            
            rate.sleep()
            
            # Check if reached target
            q_error = np.linalg.norm(data.qpos[:6] - target_q)
            if q_error < 0.05:
                print(f"Episode {episode+1}/{n_episodes}: Reached target in {step} steps")
                break
        
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode+1}/{n_episodes} episodes, {len(observations)} samples")
    
    # Convert to arrays
    observations = np.array(observations)
    actions = np.array(actions)
    targets = np.array(targets)
    
    print(f"\nData collection complete!")
    print(f"Total samples: {len(observations)}")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    metadata = {
        'n_episodes': n_episodes,
        'steps_per_episode': steps_per_episode,
        'total_samples': len(observations),
        'obs_dim': observations.shape[1],
        'action_dim': actions.shape[1],
        'timestamp': datetime.now().isoformat()
    }
    
    return observations, actions, metadata


def save_data(observations, actions, metadata, save_dir='data/raw'):
    """Save collected data."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save arrays
    np.savez(
        save_path / f'mpc_data_{timestamp}.npz',
        observations=observations,
        actions=actions
    )
    
    # Save metadata
    with open(save_path / f'mpc_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to {save_path}/")
    print(f"  - mpc_data_{timestamp}.npz")
    print(f"  - mpc_metadata_{timestamp}.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect MPC training data')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
    parser.add_argument('--render', action='store_true', help='Render simulation')
    parser.add_argument('--save-dir', type=str, default='data/raw', help='Save directory')
    
    args = parser.parse_args()
    
    # Collect data
    observations, actions, metadata = collect_data(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        render=args.render
    )
    
    # Save data
    save_data(observations, actions, metadata, save_dir=args.save_dir)
    
    print("\n✅ Data collection complete!")

