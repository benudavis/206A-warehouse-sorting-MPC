"""
DiagnosticLogger - Comprehensive data collection for manipulation tasks.

Logs:
- Robot state (joint positions, velocities)
- End-effector pose
- Object poses
- Gripper state
- Distance metrics
- Timing information
- IK/MPC convergence data
"""

import numpy as np
import mujoco
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, List, Any


class DiagnosticLogger:
    """
    Comprehensive diagnostic logger for robotic manipulation.
    
    Usage:
        logger = DiagnosticLogger(model, data, site_name="ee_site")
        
        # Add objects to track
        logger.add_tracked_object("red_box", body_id)
        
        # Log state during operation
        logger.log_state("red_box", "approach", attempt=0)
        
        # Log special events
        logger.log_ik_result("red_box", "grasp", target_pos, achieved_pos, success=True)
        logger.log_mpc_convergence("red_box", "approach", steps=150, error=0.023)
        logger.log_lift_test("red_box", attempt=0, baseline_z=0.52, new_z=0.53, success=True)
        
        # Generate comprehensive report
        logger.generate_report(output_dir="data/diagnostics")
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 site_name: str = "arm_hand_pinch",
                 gripper_joint_name: Optional[str] = "hand_left_driver_joint"):
        """
        Initialize diagnostic logger.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            site_name: End-effector site name
            gripper_joint_name: Gripper joint name for reading actual position
        """
        self.model = model
        self.data = data
        self.site_name = site_name
        self.gripper_joint_name = gripper_joint_name
        
        # Get site ID
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        
        # Get gripper joint ID if available
        self.gripper_joint_id = -1
        if gripper_joint_name:
            try:
                self.gripper_joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, gripper_joint_name
                )
            except:
                pass
        
        # Tracked objects: {name: body_id}
        self.tracked_objects: Dict[str, int] = {}
        
        # Object metadata: {name: {size, mass, etc}}
        self.object_metadata: Dict[str, Dict] = {}
        
        # Main data storage: {obj_name: {phase: [log_entries]}}
        self.logs: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        
        # Special event logs
        self.ik_results: Dict[str, List[Dict]] = defaultdict(list)
        self.mpc_convergence: Dict[str, List[Dict]] = defaultdict(list)
        self.lift_tests: Dict[str, List[Dict]] = defaultdict(list)
        self.grasp_attempts: Dict[str, List[Dict]] = defaultdict(list)
        
        # Global metrics
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Warnings and errors
        self.warnings: List[Dict] = []
        self.errors: List[Dict] = []
        
    def add_tracked_object(self, name: str, body_id: int, 
                          size: Optional[float] = None,
                          mass: Optional[float] = None,
                          initial_pos: Optional[np.ndarray] = None):
        """
        Add an object to track during operations.
        
        Args:
            name: Object name
            body_id: MuJoCo body ID
            size: Object size (half-extent for boxes)
            mass: Object mass
            initial_pos: Initial position
        """
        self.tracked_objects[name] = body_id
        
        if initial_pos is None:
            initial_pos = self.data.xpos[body_id].copy()
        
        self.object_metadata[name] = {
            'body_id': body_id,
            'size': size,
            'mass': mass,
            'initial_pos': initial_pos.tolist() if isinstance(initial_pos, np.ndarray) else initial_pos,
            'added_at': self.data.time
        }
    
    def log_state(self, obj_name: str, phase: str, attempt: int = 0,
                  extra_data: Optional[Dict] = None):
        """
        Log current state of robot, object, and gripper.
        
        Args:
            obj_name: Name of object being manipulated
            phase: Current phase (e.g., "approach", "grasp", "lift")
            attempt: Attempt number for this operation
            extra_data: Additional data to log
        """
        if obj_name not in self.tracked_objects:
            self.add_warning(f"Object {obj_name} not tracked, adding now")
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            self.add_tracked_object(obj_name, obj_id)
        
        obj_id = self.tracked_objects[obj_name]
        
        # End-effector state
        ee_pos = self.data.site_xpos[self.site_id].copy()
        ee_vel = self.data.site_xvelp[self.site_id].copy() if hasattr(self.data, 'site_xvelp') else np.zeros(3)
        
        # Object state
        obj_pos = self.data.xpos[obj_id].copy()
        obj_vel = self.data.cvel[obj_id][:3].copy() if hasattr(self.data, 'cvel') else np.zeros(3)
        obj_quat = self.data.xquat[obj_id].copy() if hasattr(self.data, 'xquat') else np.array([1, 0, 0, 0])
        
        # Robot joint state
        joint_pos = self.data.qpos[:6].copy()
        joint_vel = self.data.qvel[:6].copy()
        joint_ctrl = self.data.ctrl[:6].copy()
        
        # Gripper state
        gripper_cmd = float(self.data.ctrl[6]) if len(self.data.ctrl) > 6 else 0.0
        gripper_pos = 0.0
        if self.gripper_joint_id >= 0:
            gripper_pos = float(self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]])
        
        # Compute metrics
        distance = float(np.linalg.norm(ee_pos - obj_pos))
        distance_xy = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
        distance_z = float(abs(ee_pos[2] - obj_pos[2]))
        
        # Build log entry
        log_entry = {
            'time': float(self.data.time),
            'phase': phase,
            'attempt': attempt,
            
            # End-effector
            'ee_pos': ee_pos.tolist(),
            'ee_vel': ee_vel.tolist(),
            
            # Object
            'obj_pos': obj_pos.tolist(),
            'obj_vel': obj_vel.tolist(),
            'obj_quat': obj_quat.tolist(),
            
            # Robot
            'joint_pos': joint_pos.tolist(),
            'joint_vel': joint_vel.tolist(),
            'joint_ctrl': joint_ctrl.tolist(),
            
            # Gripper
            'gripper_cmd': gripper_cmd,
            'gripper_pos': gripper_pos,
            
            # Metrics
            'distance': distance,
            'distance_xy': distance_xy,
            'distance_z': distance_z,
        }
        
        # Add extra data if provided
        if extra_data:
            log_entry.update(extra_data)
        
        # Store
        self.logs[obj_name][phase].append(log_entry)
    
    def log_ik_result(self, obj_name: str, phase: str, 
                     target_pos: np.ndarray, 
                     achieved_pos: np.ndarray,
                     target_quat: Optional[np.ndarray] = None,
                     achieved_quat: Optional[np.ndarray] = None,
                     success: bool = False,
                     iterations: int = 0,
                     tolerance: float = 0.0):
        """Log IK solver results."""
        pos_error = float(np.linalg.norm(achieved_pos - target_pos))
        
        entry = {
            'time': float(self.data.time),
            'obj_name': obj_name,
            'phase': phase,
            'target_pos': target_pos.tolist(),
            'achieved_pos': achieved_pos.tolist(),
            'pos_error': pos_error,
            'success': success,
            'iterations': iterations,
            'tolerance': tolerance,
        }
        
        if target_quat is not None and achieved_quat is not None:
            # Quaternion error (could compute angle between them)
            entry['target_quat'] = target_quat.tolist()
            entry['achieved_quat'] = achieved_quat.tolist()
        
        self.ik_results[obj_name].append(entry)
        
        # Add warning if large error
        if pos_error > 0.05:  # 5cm
            self.add_warning(f"Large IK error for {obj_name}/{phase}: {pos_error*1000:.1f}mm")
    
    def log_mpc_convergence(self, obj_name: str, phase: str,
                           target_joints: np.ndarray,
                           achieved_joints: Optional[np.ndarray] = None,
                           steps: int = 0,
                           converged: bool = False,
                           final_error: float = 0.0,
                           tolerance: float = 0.0):
        """Log MPC convergence information."""
        if achieved_joints is None:
            achieved_joints = self.data.qpos[:6].copy()
        
        joint_error = float(np.linalg.norm(achieved_joints - target_joints))
        
        entry = {
            'time': float(self.data.time),
            'obj_name': obj_name,
            'phase': phase,
            'target_joints': target_joints.tolist(),
            'achieved_joints': achieved_joints.tolist(),
            'joint_error': joint_error,
            'steps': steps,
            'converged': converged,
            'final_error': final_error,
            'tolerance': tolerance,
        }
        
        self.mpc_convergence[obj_name].append(entry)
        
        if not converged:
            self.add_warning(f"MPC did not converge for {obj_name}/{phase} (error: {final_error:.4f})")
    
    def log_lift_test(self, obj_name: str, attempt: int,
                     baseline_z: float, new_z: float,
                     threshold: float = 0.006,
                     success: Optional[bool] = None):
        """Log lift test results."""
        lift_delta = new_z - baseline_z
        
        if success is None:
            success = lift_delta > threshold
        
        entry = {
            'time': float(self.data.time),
            'obj_name': obj_name,
            'attempt': attempt,
            'baseline_z': baseline_z,
            'new_z': new_z,
            'lift_delta': lift_delta,
            'threshold': threshold,
            'success': success,
        }
        
        self.lift_tests[obj_name].append(entry)
    
    def log_grasp_attempt(self, obj_name: str, attempt: int,
                         approach_vector: Optional[np.ndarray] = None,
                         contact_distance: float = 0.0,
                         gripper_closure: float = 0.0,
                         success: bool = False,
                         reason: str = ""):
        """Log grasp attempt details."""
        entry = {
            'time': float(self.data.time),
            'obj_name': obj_name,
            'attempt': attempt,
            'contact_distance': contact_distance,
            'gripper_closure': gripper_closure,
            'success': success,
            'reason': reason,
        }
        
        if approach_vector is not None:
            entry['approach_vector'] = approach_vector.tolist()
        
        self.grasp_attempts[obj_name].append(entry)
    
    def add_warning(self, message: str):
        """Add a warning to the log."""
        self.warnings.append({
            'time': float(self.data.time),
            'timestamp': datetime.now().isoformat(),
            'message': message
        })
        print(f"  ⚠️  {message}")
    
    def add_error(self, message: str):
        """Add an error to the log."""
        self.errors.append({
            'time': float(self.data.time),
            'timestamp': datetime.now().isoformat(),
            'message': message
        })
        print(f"  ❌ {message}")
    
    def get_summary_stats(self, obj_name: str) -> Dict[str, Any]:
        """Get summary statistics for an object."""
        if obj_name not in self.logs:
            return {}
        
        stats = {
            'object': obj_name,
            'total_phases': len(self.logs[obj_name]),
            'total_logs': sum(len(logs) for logs in self.logs[obj_name].values()),
        }
        
        # IK stats
        if obj_name in self.ik_results:
            ik_data = self.ik_results[obj_name]
            stats['ik_attempts'] = len(ik_data)
            stats['ik_success_rate'] = sum(1 for r in ik_data if r['success']) / len(ik_data) if ik_data else 0
            stats['ik_avg_error'] = np.mean([r['pos_error'] for r in ik_data]) if ik_data else 0
        
        # MPC stats
        if obj_name in self.mpc_convergence:
            mpc_data = self.mpc_convergence[obj_name]
            stats['mpc_attempts'] = len(mpc_data)
            stats['mpc_convergence_rate'] = sum(1 for r in mpc_data if r['converged']) / len(mpc_data) if mpc_data else 0
            stats['mpc_avg_steps'] = np.mean([r['steps'] for r in mpc_data]) if mpc_data else 0
        
        # Lift test stats
        if obj_name in self.lift_tests:
            lift_data = self.lift_tests[obj_name]
            stats['lift_attempts'] = len(lift_data)
            stats['lift_success_rate'] = sum(1 for r in lift_data if r['success']) / len(lift_data) if lift_data else 0
            stats['lift_avg_delta'] = np.mean([r['lift_delta'] for r in lift_data]) if lift_data else 0
        
        # Distance stats across all phases
        all_distances = []
        for phase_logs in self.logs[obj_name].values():
            all_distances.extend([log['distance'] for log in phase_logs])
        
        if all_distances:
            stats['min_distance'] = float(np.min(all_distances))
            stats['avg_distance'] = float(np.mean(all_distances))
            stats['max_distance'] = float(np.max(all_distances))
        
        return stats
    
    def save_raw_data(self, output_dir: Path):
        """Save raw log data to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main logs
        logs_file = output_dir / f"logs_{self.session_id}.npz"
        np.savez_compressed(
            logs_file,
            logs=dict(self.logs),
            tracked_objects=self.tracked_objects,
            object_metadata=self.object_metadata,
        )
        print(f"  📁 Saved raw logs: {logs_file}")
        
        # Save events
        events_file = output_dir / f"events_{self.session_id}.json"
        events_data = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'ik_results': dict(self.ik_results),
            'mpc_convergence': dict(self.mpc_convergence),
            'lift_tests': dict(self.lift_tests),
            'grasp_attempts': dict(self.grasp_attempts),
            'warnings': self.warnings,
            'errors': self.errors,
        }
        
        with open(events_file, 'w') as f:
            json.dump(events_data, f, indent=2)
        print(f"  📁 Saved events: {events_file}")
        
        # Save summary statistics
        summary_file = output_dir / f"summary_{self.session_id}.json"
        summary = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'objects': {name: self.get_summary_stats(name) for name in self.tracked_objects},
            'total_warnings': len(self.warnings),
            'total_errors': len(self.errors),
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  📁 Saved summary: {summary_file}")
        
        return logs_file, events_file, summary_file
    
    def generate_report(self, output_dir: str = "data/diagnostics",
                       show_plots: bool = True,
                       save_plots: bool = True):
        """
        Generate comprehensive diagnostic report.
        
        Args:
            output_dir: Directory to save report files
            show_plots: Whether to display plots
            save_plots: Whether to save plot images
        """
        from .plotter import DiagnosticPlotter
        from .metrics import MetricsCalculator
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"GENERATING DIAGNOSTIC REPORT")
        print(f"{'='*70}")
        
        # Save raw data
        self.save_raw_data(output_dir)
        
        # Generate plots for each object
        plotter = DiagnosticPlotter(self)
        
        for obj_name in self.tracked_objects:
            print(f"\n  Generating report for {obj_name}...")
            
            fig = plotter.create_comprehensive_report(obj_name)
            
            if save_plots:
                plot_file = output_dir / f"report_{obj_name}_{self.session_id}.png"
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"    📊 Saved plot: {plot_file}")
            
            if show_plots:
                import matplotlib.pyplot as plt
                plt.show()
            else:
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        # Generate metrics report
        metrics = MetricsCalculator(self)
        metrics_report = metrics.generate_report()
        
        metrics_file = output_dir / f"metrics_{self.session_id}.txt"
        with open(metrics_file, 'w') as f:
            f.write(metrics_report)
        print(f"\n  📊 Saved metrics report: {metrics_file}")
        
        print(f"\n{'='*70}")
        print(f"REPORT GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nAll files saved to: {output_dir}")
        
        return output_dir

