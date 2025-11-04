"""
MetricsCalculator - Compute performance metrics from diagnostic data.

Analyzes:
- Success rates
- Timing statistics
- Accuracy metrics
- Failure modes
- Performance bottlenecks
"""

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from .logger import DiagnosticLogger


class MetricsCalculator:
    """Calculate performance metrics from diagnostic data."""
    
    def __init__(self, logger: 'DiagnosticLogger'):
        """
        Initialize metrics calculator.
        
        Args:
            logger: DiagnosticLogger instance with collected data
        """
        self.logger = logger
    
    def calculate_ik_metrics(self, obj_name: str) -> Dict[str, Any]:
        """Calculate IK performance metrics."""
        if obj_name not in self.logger.ik_results or not self.logger.ik_results[obj_name]:
            return {}
        
        ik_data = self.logger.ik_results[obj_name]
        
        errors = [d['pos_error'] for d in ik_data]
        successes = [d['success'] for d in ik_data]
        iterations = [d['iterations'] for d in ik_data if d['iterations'] > 0]
        
        metrics = {
            'total_attempts': len(ik_data),
            'success_count': sum(successes),
            'success_rate': np.mean(successes) if successes else 0,
            'error_mean': np.mean(errors),
            'error_std': np.std(errors),
            'error_min': np.min(errors),
            'error_max': np.max(errors),
            'error_median': np.median(errors),
        }
        
        if iterations:
            metrics.update({
                'iterations_mean': np.mean(iterations),
                'iterations_std': np.std(iterations),
                'iterations_max': np.max(iterations),
            })
        
        # Categorize errors
        metrics['excellent_count'] = sum(1 for e in errors if e < 0.01)  # <1cm
        metrics['good_count'] = sum(1 for e in errors if 0.01 <= e < 0.03)  # 1-3cm
        metrics['marginal_count'] = sum(1 for e in errors if 0.03 <= e < 0.05)  # 3-5cm
        metrics['poor_count'] = sum(1 for e in errors if e >= 0.05)  # >5cm
        
        return metrics
    
    def calculate_mpc_metrics(self, obj_name: str) -> Dict[str, Any]:
        """Calculate MPC performance metrics."""
        if obj_name not in self.logger.mpc_convergence or not self.logger.mpc_convergence[obj_name]:
            return {}
        
        mpc_data = self.logger.mpc_convergence[obj_name]
        
        steps = [d['steps'] for d in mpc_data]
        converged = [d['converged'] for d in mpc_data]
        errors = [d['final_error'] for d in mpc_data]
        
        metrics = {
            'total_attempts': len(mpc_data),
            'converged_count': sum(converged),
            'convergence_rate': np.mean(converged) if converged else 0,
            'steps_mean': np.mean(steps),
            'steps_std': np.std(steps),
            'steps_min': np.min(steps),
            'steps_max': np.max(steps),
            'steps_median': np.median(steps),
            'error_mean': np.mean(errors),
            'error_min': np.min(errors),
            'error_max': np.max(errors),
        }
        
        # Convergence speed categories
        metrics['fast_convergence'] = sum(1 for s in steps if s < 100)  # <100 steps
        metrics['medium_convergence'] = sum(1 for s in steps if 100 <= s < 500)
        metrics['slow_convergence'] = sum(1 for s in steps if s >= 500)
        
        return metrics
    
    def calculate_lift_metrics(self, obj_name: str) -> Dict[str, Any]:
        """Calculate lift test metrics."""
        if obj_name not in self.logger.lift_tests or not self.logger.lift_tests[obj_name]:
            return {}
        
        lift_data = self.logger.lift_tests[obj_name]
        
        deltas = [d['lift_delta'] for d in lift_data]
        successes = [d['success'] for d in lift_data]
        
        metrics = {
            'total_attempts': len(lift_data),
            'success_count': sum(successes),
            'success_rate': np.mean(successes) if successes else 0,
            'delta_mean': np.mean(deltas),
            'delta_std': np.std(deltas),
            'delta_min': np.min(deltas),
            'delta_max': np.max(deltas),
            'first_success_attempt': next((i for i, s in enumerate(successes) if s), -1),
        }
        
        return metrics
    
    def calculate_distance_metrics(self, obj_name: str) -> Dict[str, Any]:
        """Calculate distance-related metrics."""
        if obj_name not in self.logger.logs:
            return {}
        
        logs = self.logger.logs[obj_name]
        
        all_distances = []
        all_distances_xy = []
        all_distances_z = []
        
        phase_metrics = {}
        
        for phase_name, phase_logs in logs.items():
            if not phase_logs:
                continue
            
            distances = [log['distance'] for log in phase_logs]
            distances_xy = [log['distance_xy'] for log in phase_logs]
            distances_z = [log['distance_z'] for log in phase_logs]
            
            all_distances.extend(distances)
            all_distances_xy.extend(distances_xy)
            all_distances_z.extend(distances_z)
            
            phase_metrics[phase_name] = {
                'min': np.min(distances),
                'mean': np.mean(distances),
                'max': np.max(distances),
                'final': distances[-1],
                'start': distances[0],
                'improvement': distances[0] - distances[-1],
            }
        
        metrics = {
            'overall_min': np.min(all_distances),
            'overall_mean': np.mean(all_distances),
            'overall_max': np.max(all_distances),
            'overall_std': np.std(all_distances),
            'xy_mean': np.mean(all_distances_xy),
            'z_mean': np.mean(all_distances_z),
            'phase_metrics': phase_metrics,
        }
        
        # Categorize final distances by phase
        good_approaches = sum(1 for pm in phase_metrics.values() if pm['final'] < 0.03)
        metrics['phases_with_good_final_distance'] = good_approaches
        
        return metrics
    
    def calculate_timing_metrics(self, obj_name: str) -> Dict[str, Any]:
        """Calculate timing and duration metrics."""
        if obj_name not in self.logger.logs:
            return {}
        
        logs = self.logger.logs[obj_name]
        
        phase_durations = {}
        total_start = float('inf')
        total_end = 0
        
        for phase_name, phase_logs in logs.items():
            if not phase_logs:
                continue
            
            start_time = phase_logs[0]['time']
            end_time = phase_logs[-1]['time']
            duration = end_time - start_time
            
            phase_durations[phase_name] = {
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'num_steps': len(phase_logs),
            }
            
            total_start = min(total_start, start_time)
            total_end = max(total_end, end_time)
        
        metrics = {
            'total_duration': total_end - total_start,
            'phase_durations': phase_durations,
            'num_phases': len(phase_durations),
        }
        
        if phase_durations:
            durations = [pd['duration'] for pd in phase_durations.values()]
            metrics['phase_duration_mean'] = np.mean(durations)
            metrics['phase_duration_std'] = np.std(durations)
            metrics['longest_phase'] = max(phase_durations.items(), key=lambda x: x[1]['duration'])[0]
            metrics['shortest_phase'] = min(phase_durations.items(), key=lambda x: x[1]['duration'])[0]
        
        return metrics
    
    def identify_failure_modes(self, obj_name: str) -> List[str]:
        """Identify potential failure modes based on collected data."""
        failures = []
        
        # Check IK failures
        if obj_name in self.logger.ik_results:
            ik_metrics = self.calculate_ik_metrics(obj_name)
            if ik_metrics.get('success_rate', 1.0) < 0.8:
                failures.append(f"Low IK success rate: {ik_metrics['success_rate']*100:.1f}%")
            if ik_metrics.get('poor_count', 0) > 0:
                failures.append(f"IK produced {ik_metrics['poor_count']} poor solutions (>5cm error)")
        
        # Check MPC convergence issues
        if obj_name in self.logger.mpc_convergence:
            mpc_metrics = self.calculate_mpc_metrics(obj_name)
            if mpc_metrics.get('convergence_rate', 1.0) < 0.8:
                failures.append(f"Low MPC convergence: {mpc_metrics['convergence_rate']*100:.1f}%")
            if mpc_metrics.get('slow_convergence', 0) > 0:
                failures.append(f"MPC slow convergence in {mpc_metrics['slow_convergence']} attempts")
        
        # Check lift test failures
        if obj_name in self.logger.lift_tests:
            lift_metrics = self.calculate_lift_metrics(obj_name)
            if lift_metrics.get('success_rate', 0) == 0:
                failures.append("ALL lift tests failed - object never grasped")
            elif lift_metrics.get('success_rate', 1.0) < 0.5:
                failures.append(f"Low lift success: {lift_metrics['success_rate']*100:.1f}%")
        
        # Check distance issues
        if obj_name in self.logger.logs:
            dist_metrics = self.calculate_distance_metrics(obj_name)
            if dist_metrics.get('overall_min', 0) > 0.05:
                failures.append(f"Never got close to object (min dist: {dist_metrics['overall_min']*1000:.1f}mm)")
        
        # Check warnings and errors
        obj_warnings = [w for w in self.logger.warnings if obj_name in w['message']]
        obj_errors = [e for e in self.logger.errors if obj_name in e['message']]
        
        if obj_errors:
            failures.append(f"{len(obj_errors)} errors logged")
        if len(obj_warnings) > 5:
            failures.append(f"{len(obj_warnings)} warnings logged")
        
        return failures
    
    def generate_report(self) -> str:
        """Generate comprehensive text metrics report."""
        lines = []
        lines.append("="*80)
        lines.append("DIAGNOSTIC METRICS REPORT")
        lines.append("="*80)
        lines.append(f"Session ID: {self.logger.session_id}")
        lines.append(f"Start Time: {self.logger.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        for obj_name in self.logger.tracked_objects:
            lines.append("="*80)
            lines.append(f"OBJECT: {obj_name}")
            lines.append("="*80)
            lines.append("")
            
            # IK Metrics
            ik_metrics = self.calculate_ik_metrics(obj_name)
            if ik_metrics:
                lines.append("--- IK PERFORMANCE ---")
                lines.append(f"  Total Attempts:     {ik_metrics['total_attempts']}")
                lines.append(f"  Success Rate:       {ik_metrics['success_rate']*100:.1f}%")
                lines.append(f"  Error (mean ± std): {ik_metrics['error_mean']*1000:.1f} ± {ik_metrics['error_std']*1000:.1f} mm")
                lines.append(f"  Error (min/max):    {ik_metrics['error_min']*1000:.1f} / {ik_metrics['error_max']*1000:.1f} mm")
                lines.append(f"  Error (median):     {ik_metrics['error_median']*1000:.1f} mm")
                if 'iterations_mean' in ik_metrics:
                    lines.append(f"  Iterations (avg):   {ik_metrics['iterations_mean']:.0f}")
                lines.append(f"  Excellent (<1cm):   {ik_metrics['excellent_count']}")
                lines.append(f"  Good (1-3cm):       {ik_metrics['good_count']}")
                lines.append(f"  Marginal (3-5cm):   {ik_metrics['marginal_count']}")
                lines.append(f"  Poor (>5cm):        {ik_metrics['poor_count']}")
                lines.append("")
            
            # MPC Metrics
            mpc_metrics = self.calculate_mpc_metrics(obj_name)
            if mpc_metrics:
                lines.append("--- MPC PERFORMANCE ---")
                lines.append(f"  Total Attempts:     {mpc_metrics['total_attempts']}")
                lines.append(f"  Convergence Rate:   {mpc_metrics['convergence_rate']*100:.1f}%")
                lines.append(f"  Steps (mean ± std): {mpc_metrics['steps_mean']:.0f} ± {mpc_metrics['steps_std']:.0f}")
                lines.append(f"  Steps (min/max):    {mpc_metrics['steps_min']} / {mpc_metrics['steps_max']}")
                lines.append(f"  Steps (median):     {mpc_metrics['steps_median']:.0f}")
                lines.append(f"  Fast (<100):        {mpc_metrics['fast_convergence']}")
                lines.append(f"  Medium (100-500):   {mpc_metrics['medium_convergence']}")
                lines.append(f"  Slow (>500):        {mpc_metrics['slow_convergence']}")
                lines.append("")
            
            # Lift Metrics
            lift_metrics = self.calculate_lift_metrics(obj_name)
            if lift_metrics:
                lines.append("--- LIFT TEST RESULTS ---")
                lines.append(f"  Total Attempts:     {lift_metrics['total_attempts']}")
                lines.append(f"  Success Rate:       {lift_metrics['success_rate']*100:.1f}%")
                lines.append(f"  Delta (mean ± std): {lift_metrics['delta_mean']*1000:.1f} ± {lift_metrics['delta_std']*1000:.1f} mm")
                lines.append(f"  Delta (min/max):    {lift_metrics['delta_min']*1000:.1f} / {lift_metrics['delta_max']*1000:.1f} mm")
                if lift_metrics['first_success_attempt'] >= 0:
                    lines.append(f"  First Success:      Attempt {lift_metrics['first_success_attempt']}")
                else:
                    lines.append(f"  First Success:      NEVER")
                lines.append("")
            
            # Distance Metrics
            dist_metrics = self.calculate_distance_metrics(obj_name)
            if dist_metrics:
                lines.append("--- DISTANCE METRICS ---")
                lines.append(f"  Overall Min:        {dist_metrics['overall_min']*1000:.1f} mm")
                lines.append(f"  Overall Mean:       {dist_metrics['overall_mean']*1000:.1f} mm")
                lines.append(f"  Overall Max:        {dist_metrics['overall_max']*1000:.1f} mm")
                lines.append(f"  Overall Std:        {dist_metrics['overall_std']*1000:.1f} mm")
                lines.append(f"  XY Mean:            {dist_metrics['xy_mean']*1000:.1f} mm")
                lines.append(f"  Z Mean:             {dist_metrics['z_mean']*1000:.1f} mm")
                lines.append(f"  Good Final Dist:    {dist_metrics.get('phases_with_good_final_distance', 0)} phases")
                
                if dist_metrics.get('phase_metrics'):
                    lines.append("\n  Per-Phase Distance:")
                    for phase, pm in dist_metrics['phase_metrics'].items():
                        lines.append(f"    {phase:20s}: {pm['start']*1000:6.1f} → {pm['final']*1000:6.1f} mm (Δ {pm['improvement']*1000:+6.1f})")
                lines.append("")
            
            # Timing Metrics
            timing_metrics = self.calculate_timing_metrics(obj_name)
            if timing_metrics:
                lines.append("--- TIMING ---")
                lines.append(f"  Total Duration:     {timing_metrics['total_duration']:.2f} s")
                lines.append(f"  Num Phases:         {timing_metrics['num_phases']}")
                if 'phase_duration_mean' in timing_metrics:
                    lines.append(f"  Phase Avg Duration: {timing_metrics['phase_duration_mean']:.2f} s")
                    lines.append(f"  Longest Phase:      {timing_metrics['longest_phase']}")
                    lines.append(f"  Shortest Phase:     {timing_metrics['shortest_phase']}")
                lines.append("")
            
            # Failure Modes
            failures = self.identify_failure_modes(obj_name)
            if failures:
                lines.append("--- IDENTIFIED ISSUES ---")
                for failure in failures:
                    lines.append(f"  ❌ {failure}")
                lines.append("")
            else:
                lines.append("--- IDENTIFIED ISSUES ---")
                lines.append("  ✅ No major issues detected")
                lines.append("")
        
        # Global summary
        lines.append("="*80)
        lines.append("GLOBAL SUMMARY")
        lines.append("="*80)
        lines.append(f"Total Objects Tracked: {len(self.logger.tracked_objects)}")
        lines.append(f"Total Warnings:        {len(self.logger.warnings)}")
        lines.append(f"Total Errors:          {len(self.logger.errors)}")
        lines.append("")
        
        if self.logger.warnings:
            lines.append("Recent Warnings:")
            for w in self.logger.warnings[-5:]:
                lines.append(f"  [{w['time']:.2f}s] {w['message']}")
            lines.append("")
        
        if self.logger.errors:
            lines.append("Recent Errors:")
            for e in self.logger.errors[-5:]:
                lines.append(f"  [{e['time']:.2f}s] {e['message']}")
            lines.append("")
        
        lines.append("="*80)
        
        return '\n'.join(lines)

