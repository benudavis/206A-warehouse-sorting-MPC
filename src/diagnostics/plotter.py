"""
DiagnosticPlotter - Visualization for diagnostic data.

Creates comprehensive plots showing:
- Trajectories (3D and 2D projections)
- Distance metrics over time
- Gripper behavior
- Joint motions
- Phase-by-phase analysis
- Success/failure summaries
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import DiagnosticLogger


class DiagnosticPlotter:
    """Creates diagnostic visualizations from logger data."""
    
    def __init__(self, logger: 'DiagnosticLogger'):
        """
        Initialize plotter with logger data.
        
        Args:
            logger: DiagnosticLogger instance with collected data
        """
        self.logger = logger
    
    def create_comprehensive_report(self, obj_name: str):
        """
        Create comprehensive diagnostic report for an object.
        
        Returns matplotlib figure with multiple subplots.
        """
        if obj_name not in self.logger.tracked_objects:
            raise ValueError(f"Object {obj_name} not tracked")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: 3D Trajectory
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self._plot_3d_trajectory(ax1, obj_name)
        
        # Plot 2: XY Plane View
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_xy_plane(ax2, obj_name)
        
        # Plot 3: XZ Plane View
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_xz_plane(ax3, obj_name)
        
        # Plot 4: Distance over Time
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_distance_over_time(ax4, obj_name)
        
        # Plot 5: Z Height Evolution
        ax5 = fig.add_subplot(gs[1, 3])
        self._plot_z_height(ax5, obj_name)
        
        # Plot 6: Gripper Behavior
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_gripper(ax6, obj_name)
        
        # Plot 7: Joint Trajectories
        ax7 = fig.add_subplot(gs[2, 1:3])
        self._plot_joint_trajectories(ax7, obj_name)
        
        # Plot 8: Lift Test Results
        ax8 = fig.add_subplot(gs[2, 3])
        self._plot_lift_tests(ax8, obj_name)
        
        # Plot 9: Distance Statistics by Phase
        ax9 = fig.add_subplot(gs[3, 0])
        self._plot_distance_by_phase(ax9, obj_name)
        
        # Plot 10: IK Convergence
        ax10 = fig.add_subplot(gs[3, 1])
        self._plot_ik_convergence(ax10, obj_name)
        
        # Plot 11: MPC Convergence
        ax11 = fig.add_subplot(gs[3, 2])
        self._plot_mpc_convergence(ax11, obj_name)
        
        # Plot 12: Summary Text
        ax12 = fig.add_subplot(gs[3, 3])
        self._plot_summary_text(ax12, obj_name)
        
        # Overall title
        fig.suptitle(f'Diagnostic Report: {obj_name} (Session: {self.logger.session_id})', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def _plot_3d_trajectory(self, ax, obj_name):
        """Plot 3D trajectory of end-effector and object."""
        logs = self.logger.logs[obj_name]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))
        
        for (phase_name, phase_logs), color in zip(logs.items(), colors):
            if not phase_logs:
                continue
            
            ee_traj = np.array([log['ee_pos'] for log in phase_logs])
            obj_traj = np.array([log['obj_pos'] for log in phase_logs])
            
            # Plot EE trajectory
            ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
                   label=f'EE: {phase_name}', color=color, linewidth=2, alpha=0.7)
            
            # Mark start and end
            ax.scatter(ee_traj[0, 0], ee_traj[0, 1], ee_traj[0, 2],
                      color=color, s=50, marker='o', alpha=0.5)
            ax.scatter(ee_traj[-1, 0], ee_traj[-1, 1], ee_traj[-1, 2],
                      color=color, s=50, marker='s', alpha=0.5)
            
            # Show object position (first occurrence)
            if phase_name == list(logs.keys())[0]:
                ax.scatter(obj_traj[0, 0], obj_traj[0, 1], obj_traj[0, 2],
                          c='red', s=200, marker='*', label='Object', 
                          edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title('3D Trajectory: End-Effector Path', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_xy_plane(self, ax, obj_name):
        """Plot top-down XY view."""
        logs = self.logger.logs[obj_name]
        
        for phase_name, phase_logs in logs.items():
            if not phase_logs:
                continue
            
            ee_xy = np.array([[log['ee_pos'][0], log['ee_pos'][1]] for log in phase_logs])
            obj_xy = np.array([[log['obj_pos'][0], log['obj_pos'][1]] for log in phase_logs])
            
            ax.plot(ee_xy[:, 0], ee_xy[:, 1], label=phase_name, marker='.', markersize=3, alpha=0.7)
            
            # Object position
            if phase_name == list(logs.keys())[0]:
                ax.scatter(obj_xy[0, 0], obj_xy[0, 1], c='red', s=150, marker='*',
                          edgecolors='black', linewidths=1.5, zorder=10)
        
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title('XY Plane (Top View)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _plot_xz_plane(self, ax, obj_name):
        """Plot side XZ view."""
        logs = self.logger.logs[obj_name]
        
        for phase_name, phase_logs in logs.items():
            if not phase_logs:
                continue
            
            ee_xz = np.array([[log['ee_pos'][0], log['ee_pos'][2]] for log in phase_logs])
            obj_xz = np.array([[log['obj_pos'][0], log['obj_pos'][2]] for log in phase_logs])
            
            ax.plot(ee_xz[:, 0], ee_xz[:, 1], label=phase_name, marker='.', markersize=3, alpha=0.7)
            
            if phase_name == list(logs.keys())[0]:
                ax.scatter(obj_xz[0, 0], obj_xz[0, 1], c='red', s=150, marker='*',
                          edgecolors='black', linewidths=1.5, zorder=10)
        
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Z (m)', fontsize=9)
        ax.set_title('XZ Plane (Side View)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    def _plot_distance_over_time(self, ax, obj_name):
        """Plot distance between EE and object over time."""
        logs = self.logger.logs[obj_name]
        
        for phase_name, phase_logs in logs.items():
            if not phase_logs:
                continue
            
            times = [log['time'] for log in phase_logs]
            distances = [log['distance'] * 1000 for log in phase_logs]  # mm
            
            ax.plot(times, distances, label=phase_name, marker='.', markersize=3, alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Distance (mm)', fontsize=9)
        ax.set_title('EE-Object Distance vs Time', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # Add threshold line for "good grasp distance"
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Good (<30mm)')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Marginal (<50mm)')
    
    def _plot_z_height(self, ax, obj_name):
        """Plot Z height evolution."""
        logs = self.logger.logs[obj_name]
        
        for phase_name, phase_logs in logs.items():
            if not phase_logs:
                continue
            
            times = [log['time'] for log in phase_logs]
            ee_z = [log['ee_pos'][2] for log in phase_logs]
            obj_z = [log['obj_pos'][2] for log in phase_logs]
            
            ax.plot(times, ee_z, label=f'{phase_name} (EE)', linewidth=2, alpha=0.7)
            ax.plot(times, obj_z, label=f'{phase_name} (Obj)', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Z Height (m)', fontsize=9)
        ax.set_title('Z Height Evolution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    def _plot_gripper(self, ax, obj_name):
        """Plot gripper command and actual position."""
        logs = self.logger.logs[obj_name]
        
        all_times = []
        all_cmd = []
        all_pos = []
        
        for phase_logs in logs.values():
            for log in phase_logs:
                all_times.append(log['time'])
                all_cmd.append(log['gripper_cmd'])
                all_pos.append(log['gripper_pos'])
        
        if all_times:
            ax.plot(all_times, all_cmd, label='Command', linewidth=2, alpha=0.7)
            # Scale gripper position to be visible
            ax.plot(all_times, np.array(all_pos) * 500, label='Position (scaled 500x)', 
                   linewidth=2, alpha=0.7, linestyle='--')
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_title('Gripper Control', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _plot_joint_trajectories(self, ax, obj_name):
        """Plot all joint positions over time."""
        logs = self.logger.logs[obj_name]
        
        all_times = []
        all_joints = []
        
        for phase_logs in logs.values():
            for log in phase_logs:
                all_times.append(log['time'])
                all_joints.append(log['joint_pos'])
        
        if all_joints:
            all_joints = np.array(all_joints)
            joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 
                          'Wrist 1', 'Wrist 2', 'Wrist 3']
            
            for i in range(6):
                ax.plot(all_times, all_joints[:, i], label=joint_names[i], alpha=0.7)
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Joint Position (rad)', fontsize=9)
            ax.set_title('Joint Trajectories', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
    
    def _plot_lift_tests(self, ax, obj_name):
        """Plot lift test results."""
        if obj_name not in self.logger.lift_tests or not self.logger.lift_tests[obj_name]:
            ax.text(0.5, 0.5, 'No lift test data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('Lift Test Results', fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        lift_data = self.logger.lift_tests[obj_name]
        attempts = [d['attempt'] for d in lift_data]
        deltas = [d['lift_delta'] * 1000 for d in lift_data]  # mm
        successes = [d['success'] for d in lift_data]
        
        colors = ['green' if s else 'red' for s in successes]
        bars = ax.bar(attempts, deltas, color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold line
        threshold = lift_data[0]['threshold'] * 1000  # mm
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, 
                  label=f'Success threshold ({threshold:.1f}mm)')
        
        # Add value labels on bars
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{delta:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Attempt', fontsize=9)
        ax.set_ylabel('Lift Delta (mm)', fontsize=9)
        ax.set_title('Lift Test Results', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_distance_by_phase(self, ax, obj_name):
        """Plot distance statistics by phase."""
        logs = self.logger.logs[obj_name]
        
        phase_stats = {}
        for phase_name, phase_logs in logs.items():
            if phase_logs:
                distances = [log['distance'] * 1000 for log in phase_logs]
                phase_stats[phase_name] = {
                    'min': np.min(distances),
                    'mean': np.mean(distances),
                    'max': np.max(distances),
                    'final': distances[-1]
                }
        
        if not phase_stats:
            ax.axis('off')
            return
        
        phase_names = list(phase_stats.keys())
        means = [phase_stats[p]['mean'] for p in phase_names]
        finals = [phase_stats[p]['final'] for p in phase_names]
        
        x = np.arange(len(phase_names))
        width = 0.35
        
        ax.bar(x - width/2, means, width, label='Mean', alpha=0.7)
        ax.bar(x + width/2, finals, width, label='Final', alpha=0.7)
        
        ax.set_xlabel('Phase', fontsize=9)
        ax.set_ylabel('Distance (mm)', fontsize=9)
        ax.set_title('Distance Stats by Phase', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_ik_convergence(self, ax, obj_name):
        """Plot IK convergence statistics."""
        if obj_name not in self.logger.ik_results or not self.logger.ik_results[obj_name]:
            ax.text(0.5, 0.5, 'No IK data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('IK Convergence', fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        ik_data = self.logger.ik_results[obj_name]
        phases = [d['phase'] for d in ik_data]
        errors = [d['pos_error'] * 1000 for d in ik_data]  # mm
        successes = [d['success'] for d in ik_data]
        
        colors = ['green' if s else 'red' for s in successes]
        
        x = np.arange(len(phases))
        bars = ax.bar(x, errors, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Phase', fontsize=9)
        ax.set_ylabel('Position Error (mm)', fontsize=9)
        ax.set_title('IK Convergence', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add success/fail legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Success'),
                          Patch(facecolor='red', alpha=0.7, label='Failed')]
        ax.legend(handles=legend_elements, fontsize=8)
    
    def _plot_mpc_convergence(self, ax, obj_name):
        """Plot MPC convergence statistics."""
        if obj_name not in self.logger.mpc_convergence or not self.logger.mpc_convergence[obj_name]:
            ax.text(0.5, 0.5, 'No MPC data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title('MPC Convergence', fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        mpc_data = self.logger.mpc_convergence[obj_name]
        phases = [d['phase'] for d in mpc_data]
        steps = [d['steps'] for d in mpc_data]
        converged = [d['converged'] for d in mpc_data]
        
        colors = ['green' if c else 'red' for c in converged]
        
        x = np.arange(len(phases))
        bars = ax.bar(x, steps, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, step in zip(bars, steps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{step}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Phase', fontsize=9)
        ax.set_ylabel('Steps', fontsize=9)
        ax.set_title('MPC Convergence', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Converged'),
                          Patch(facecolor='red', alpha=0.7, label='Did not converge')]
        ax.legend(handles=legend_elements, fontsize=8)
    
    def _plot_summary_text(self, ax, obj_name):
        """Plot summary statistics as text."""
        ax.axis('off')
        
        stats = self.logger.get_summary_stats(obj_name)
        
        summary_lines = [
            f"Object: {obj_name}",
            f"Session: {self.logger.session_id}",
            "",
            "=== SUMMARY ===",
            f"Total Phases: {stats.get('total_phases', 0)}",
            f"Total Log Entries: {stats.get('total_logs', 0)}",
            "",
        ]
        
        if 'ik_attempts' in stats:
            summary_lines.extend([
                "=== IK ===",
                f"Attempts: {stats['ik_attempts']}",
                f"Success Rate: {stats['ik_success_rate']*100:.1f}%",
                f"Avg Error: {stats['ik_avg_error']*1000:.1f}mm",
                "",
            ])
        
        if 'mpc_attempts' in stats:
            summary_lines.extend([
                "=== MPC ===",
                f"Attempts: {stats['mpc_attempts']}",
                f"Convergence: {stats['mpc_convergence_rate']*100:.1f}%",
                f"Avg Steps: {stats['mpc_avg_steps']:.0f}",
                "",
            ])
        
        if 'lift_attempts' in stats:
            summary_lines.extend([
                "=== LIFT TESTS ===",
                f"Attempts: {stats['lift_attempts']}",
                f"Success Rate: {stats['lift_success_rate']*100:.1f}%",
                f"Avg Delta: {stats['lift_avg_delta']*1000:.1f}mm",
                "",
            ])
        
        if 'min_distance' in stats:
            summary_lines.extend([
                "=== DISTANCES ===",
                f"Min: {stats['min_distance']*1000:.1f}mm",
                f"Avg: {stats['avg_distance']*1000:.1f}mm",
                f"Max: {stats['max_distance']*1000:.1f}mm",
            ])
        
        summary_text = '\n'.join(summary_lines)
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_title('Summary Statistics', fontsize=10, fontweight='bold')

