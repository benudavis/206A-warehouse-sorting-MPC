from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import numpy as np
from planning.ik import IKPlanner
from planning.mpc_controller import MPCController

class UR7e_CubeGrasp(Node):

    def __init__(self):

        super().__init__('cube_grasp')

        # Make sure this topic matches your transform node publisher
        self.cube_pub = self.create_subscription(
            PointStamped,
            '/cube_pose_in_base',   # check topic alignment with transform_cube_pose.py
            self.cube_callback,
            1
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None

        # IK planner node (uses MoveIt services)
        self.ik_planner = IKPlanner()

        # MPC controller for trajectory planning with obstacle avoidance
        self.mpc = MPCController(n_joints=6, horizon=10, dt=0.05)

        # Entries should be either JointState or the string 'toggle_grip'
        self.job_queue = []

    def joint_state_callback(self, msg: JointState):

        self.joint_state = msg

    def cube_callback(self, cube_pose: PointStamped):

        # Only process the first cube pose
        if self.cube_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        self.cube_pose = cube_pose

        self.get_logger().info(
            f"Received cube pose in base_link: "
            f"({cube_pose.point.x:.3f}, {cube_pose.point.y:.3f}, {cube_pose.point.z:.3f})"
        )

        # -----------------------------------------------------------
        # Build the job queue of JointStates and 'toggle_grip' steps
        # -----------------------------------------------------------

        # Base cube position (in base_link frame)
        cx = cube_pose.point.x
        cy = cube_pose.point.y
        cz = cube_pose.point.z

        # 1) Move to Pre-Grasp Position (gripper above the cube)
        # Offsets:
        #   x offset: 0.0
        #   y offset: -0.035
        #   z offset: +0.185
        pre_x = cx + 0.0
        pre_y = cy - 0.035
        pre_z = cz + 0.185
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("IK failed for pre-grasp pose.")
            return
        self.job_queue.append(pre_grasp_js)

        # 2) Move to Grasp Position (lower the gripper to the cube)
        # DO NOT CHANGE z offset lower than +0.16
        grasp_x = cx + 0.0
        grasp_y = cy - 0.035
        grasp_z = cz + 0.16
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("IK failed for grasp pose.")
            return
        self.job_queue.append(grasp_js)

        # 3) Close the gripper
        self.job_queue.append('toggle_grip')

        # 4) Move back to Pre-Grasp Position (lift the block)
        self.job_queue.append(pre_grasp_js)

        # 5) Move to release Position (0.4m on other side along x)
        rel_x = cx + 0.4
        rel_y = cy - 0.035
        rel_z = cz + 0.185
        release_js = self.ik_planner.compute_ik(self.joint_state, rel_x, rel_y, rel_z)
        if release_js is None:
            self.get_logger().error("IK failed for release pose.")
            return
        self.job_queue.append(release_js)

        # 6) Release the gripper
        self.job_queue.append('toggle_grip')

        # Start executing the queue
        self.execute_jobs()

    def execute_jobs(self):

        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot run MPC.")
                return
            
            traj = self._plan_with_mpc(self.joint_state, next_job)
            if traj is None:
                self.get_logger().error("MPC failed to plan trajectory")
                return
            self.get_logger().info("MPC planned trajectory")
            self._execute_joint_trajectory(traj)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next jobplan_to_joints

    def _plan_with_mpc(self, current_js: JointState, target_js: JointState) -> JointTrajectory:
        """
        Plan a trajectory from current joint state to target using MPC.
        
        Args:
            current_js: Current joint state
            target_js: Target joint state
            
        Returns:
            JointTrajectory message ready to execute
        """
        # Extract joint positions, ensuring consistent ordering
        name_to_index = {name: i for i, name in enumerate(current_js.name)}
        
        try:
            q_current = np.array(
                [current_js.position[name_to_index[name]] for name in target_js.name],
                dtype=float,
            )
        except KeyError as e:
            self.get_logger().error(f"Joint name mismatch: {e}")
            return None
        
        q_target = np.array(target_js.position, dtype=float)
        
        # Build current state [q, dq] - we don't use velocities, set to zero
        dq_current = np.zeros_like(q_current)
        current_state = np.concatenate([q_current, dq_current])
        
        # Solve MPC
        q_next, q_traj = self.mpc.compute_control(current_state, q_target)
        
        # Convert MPC trajectory to JointTrajectory format
        jt = JointTrajectory()
        jt.joint_names = list(target_js.name)
        jt.header.stamp = self.get_clock().now().to_msg()
        jt.header.frame_id = "base_link"
        
        # Add trajectory points with timing
        from builtin_interfaces.msg import Duration
        for k in range(q_traj.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = q_traj[k].tolist()
            
            # Time from start based on MPC dt
            t = float(k) * self.mpc.dt
            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
            
            jt.points.append(pt)
        
        return jt

    def _toggle_gripper(self):

        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

    def _execute_joint_trajectory(self, joint_traj):

        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):

        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')

def main(args=None):

    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
