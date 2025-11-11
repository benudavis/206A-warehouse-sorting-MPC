#!/usr/bin/env python3
"""
Evaluate trained imitator vs MPC controller
Compares performance on reaching random targets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
from loop_rate_limiters import RateLimiter
import matplotlib.pyplot as plt

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


def evaluate_controller(controller, model, data, state_extractor, target_q, 
                       max_steps=400, controller_type='MPC'):  # More steps for convergence
    """
    Evaluate a controller on reaching a target.
    
    Returns:
        success: Whether target was reached
        steps: Number of steps taken
        trajectory: Joint trajectory
        errors: Position errors over time
    """
    # Reset
    initial_q = generate_random_target()
    data.qpos[:6] = initial_q
    data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    
    trajectory = [data.qpos[:6].copy()]
    errors = []
    ctrl_rate = 0.01
    
    for step in range(max_steps):
        # Get observation
        robot_state = state_extractor.get_robot_state()
        observation = state_extractor.get_observation_vector()
        
        # Compute action
        try:
            if controller_type == 'MPC':
                action, _ = controller.compute_control(robot_state, target_q)
            else:  # Learned
                action = controller.predict(observation)
            
            data.ctrl[:6] = action
        except Exception as e:
            print(f"Controller failed: {e}")
            return False, step, trajectory, errors
        
        # Step simulation
        mujoco.mj_step(model, data, nstep=int(ctrl_rate / model.opt.timestep))
        
        # Record
        trajectory.append(data.qpos[:6].copy())
        error = np.linalg.norm(data.qpos[:6] - target_q)
        errors.append(error)
        
        # Check success
        if error < 0.05:
            return True, step + 1, trajectory, errors
    
    return False, max_steps, trajectory, errors


def compare_controllers(mpc_controller, learned_controller, n_trials=20):
    """
    Compare MPC and learned controller performance.
    
    Returns:
        results: Dict with comparison metrics
    """
    print("Loading simulation...")
    model, data = load_simulation()
    state_extractor = SimulationState(model, data)
    
    mpc_results = {'success': [], 'steps': [], 'errors': []}
    learned_results = {'success': [], 'steps': [], 'errors': []}
    
    print(f"\nEvaluating on {n_trials} random targets...")
    
    for trial in range(n_trials):
        target = generate_random_target()
        
        # Evaluate MPC
        success, steps, traj, errors = evaluate_controller(
            mpc_controller, model, data, state_extractor, target, controller_type='MPC'
        )
        mpc_results['success'].append(success)
        mpc_results['steps'].append(steps)
        mpc_results['errors'].append(errors)
        
        # Evaluate learned
        success, steps, traj, errors = evaluate_controller(
            learned_controller, model, data, state_extractor, target, controller_type='Learned'
        )
        learned_results['success'].append(success)
        learned_results['steps'].append(steps)
        learned_results['errors'].append(errors)
        
        if (trial + 1) % 5 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")
    
    # Compute statistics
    results = {
        'MPC': {
            'success_rate': np.mean(mpc_results['success']) * 100,
            'avg_steps': np.mean([s for s, success in zip(mpc_results['steps'], mpc_results['success']) if success]),
            'avg_final_error': np.mean([e[-1] if len(e) > 0 else 1.0 for e in mpc_results['errors']]),
        },
        'Learned': {
            'success_rate': np.mean(learned_results['success']) * 100,
            'avg_steps': np.mean([s for s, success in zip(learned_results['steps'], learned_results['success']) if success]),
            'avg_final_error': np.mean([e[-1] if len(e) > 0 else 1.0 for e in learned_results['errors']]),
        },
        'raw_mpc': mpc_results,
        'raw_learned': learned_results
    }
    
    return results


def plot_comparison(results, save_path=None):
    """Plot comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Success rates
    axes[0].bar(['MPC', 'Learned'], 
               [results['MPC']['success_rate'], results['Learned']['success_rate']],
               color=['blue', 'orange'])
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3)
    
    # Average steps
    axes[1].bar(['MPC', 'Learned'], 
               [results['MPC']['avg_steps'], results['Learned']['avg_steps']],
               color=['blue', 'orange'])
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Average Steps to Reach Target')
    axes[1].grid(True, alpha=0.3)
    
    # Error trajectories (sample)
    mpc_errors = results['raw_mpc']['errors'][:5]
    learned_errors = results['raw_learned']['errors'][:5]
    
    for i, errors in enumerate(mpc_errors):
        axes[2].plot(errors, 'b-', alpha=0.3, linewidth=1)
    for i, errors in enumerate(learned_errors):
        axes[2].plot(errors, 'orange', alpha=0.3, linewidth=1)
    
    axes[2].plot([], [], 'b-', label='MPC')
    axes[2].plot([], [], 'orange', label='Learned')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Position Error (rad)')
    axes[2].set_title('Error Trajectories (Sample)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MPC imitator')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--trials', type=int, default=20, help='Number of evaluation trials')
    parser.add_argument('--save-dir', type=str, default='data/processed', help='Save directory')
    
    args = parser.parse_args()
    
    # Load learned model
    print("Loading learned model...")
    learned_controller = MPCImitator()
    learned_controller.load(args.model)
    
    # Initialize MPC (with tuned parameters)
    print("Initializing MPC controller...")
    mpc_controller = MPCController(n_joints=6, horizon=30, dt=0.01)  # Use same settings as training
    
    # Compare
    results = compare_controllers(mpc_controller, learned_controller, n_trials=args.trials)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nMPC Controller:")
    print(f"  Success Rate: {results['MPC']['success_rate']:.1f}%")
    print(f"  Avg Steps: {results['MPC']['avg_steps']:.1f}")
    print(f"  Avg Final Error: {results['MPC']['avg_final_error']:.4f} rad")
    
    print(f"\nLearned Controller:")
    print(f"  Success Rate: {results['Learned']['success_rate']:.1f}%")
    print(f"  Avg Steps: {results['Learned']['avg_steps']:.1f}")
    print(f"  Avg Final Error: {results['Learned']['avg_final_error']:.4f} rad")
    
    # Performance ratio
    if results['MPC']['success_rate'] > 0:
        ratio = results['Learned']['success_rate'] / results['MPC']['success_rate'] * 100
        print(f"\nLearned achieves {ratio:.1f}% of MPC performance")
    
    # Plot
    save_path = Path(args.save_dir) / 'evaluation_results.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, save_path=save_path)
    
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

