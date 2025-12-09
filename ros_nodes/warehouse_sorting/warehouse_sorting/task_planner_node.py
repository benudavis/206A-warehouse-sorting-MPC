#!/usr/bin/env python3
"""
Task Planner Node - Complete Pick & Place Automation

Implements full pick-place cycle for warehouse sorting using MoveIt IK:
- Red boxes → red basket
- Blue boxes → blue basket
- Uses MoveIt services for IK and planning (like the working example)
"""

import numpy as np
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import String
from std_srvs.srv import Trigger
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration


class TaskPlannerNode(Node):
    """Automated pick-and-place task sequencer using MoveIt."""

    def __init__(self):
        super().__init__('task_planner')
        
        # Publishers
        self.gripper_event_pub = self.create_publisher(
            String,
            '/gripper_events',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/task_status',
            10
        )
        
        # Trajectory action client
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Gripper service client
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')
        
        # State
        self.joint_state = None
        self.job_queue = []  # Queue of JointState or 'toggle_grip'
        self.current_box_name = None
        
        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Box definitions
        self.boxes = [
            {"name": "red_1", "pos": [0.35, -0.28, 0.52], "size": 0.030, "color": "red"},
            {"name": "red_2", "pos": [0.40, -0.32, 0.52], "size": 0.028, "color": "red"},
            {"name": "red_3", "pos": [0.45, -0.28, 0.52], "size": 0.032, "color": "red"},
            {"name": "blue_1", "pos": [0.35, -0.42, 0.52], "size": 0.030, "color": "blue"},
            {"name": "blue_2", "pos": [0.40, -0.38, 0.52], "size": 0.028, "color": "blue"},
            {"name": "blue_3", "pos": [0.45, -0.42, 0.52], "size": 0.032, "color": "blue"},
        ]
        
        # Basket positions
        self.baskets = {
            "red": [-0.45, -0.10, 0.48],
            "blue": [-0.45, -0.60, 0.48],
        }
        
        # MoveIt service clients
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        # Wait for services
        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')
        
        # Start automation after delay
        self.create_timer(5.0, self.start_automation)
        
        self.get_logger().info('✓ Task Planner ready - will start automation in 5 seconds')

    def joint_state_callback(self, msg):
        """Update current joint state."""
        if len(msg.position) >= 6:
            self.joint_state = msg

    def start_automation(self):
        """Start the automated pick-place sequence."""
        if self.joint_state is None:
            self.get_logger().warn("No joint state yet, waiting...")
            return
        
        self.get_logger().info('🚀 Starting automated pick-and-place sequence!')
        
        # Run automation in a thread to avoid blocking
        thread = threading.Thread(target=self.run_pick_place_sequence)
        thread.daemon = True
        thread.start()

    def run_pick_place_sequence(self):
        """Main pick-place automation loop - builds job queue and executes."""
        time.sleep(2)  # Let system stabilize
        
        # Process each box
        for idx, box in enumerate(self.boxes):
            box_name = box["name"]
            box_color = box["color"]
            box_pos = np.array(box["pos"])
            box_size = box["size"]
            basket_pos = np.array(self.baskets[box_color])
            
            self.current_box_name = box_name
            
            self.get_logger().info(f'\n{"="*60}')
            self.get_logger().info(f'[{idx+1}/{len(self.boxes)}] {box_name} ({box_color}) → {box_color} basket')
            self.get_logger().info(f'{"="*60}')
            
            # Build job queue for this box
            self.job_queue = []
            
            # 1. Move above box (pre-grasp)
            cx, cy, cz = box_pos
            pre_x = cx + 0.0
            pre_y = cy - 0.035  # Offset like in working example
            pre_z = cz + 0.185  # ~18.5cm above
            
            self.get_logger().info(f'  Step 1: Computing IK for pre-grasp above {box_name}...')
            pre_grasp_js = self.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
            if pre_grasp_js is None:
                self.get_logger().error(f"IK failed for pre-grasp pose of {box_name}")
                continue
            self.job_queue.append(pre_grasp_js)
            
            # 2. Lower to grasp position
            grasp_x = cx + 0.0
            grasp_y = cy - 0.035
            grasp_z = cz + 0.16  # DO NOT CHANGE lower than +0.16 (from working example)
            
            self.get_logger().info(f'  Step 2: Computing IK for grasp position...')
            grasp_js = self.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
            if grasp_js is None:
                self.get_logger().error(f"IK failed for grasp pose of {box_name}")
                continue
            self.job_queue.append(grasp_js)
            
            # 3. Close gripper
            self.job_queue.append('toggle_grip')
            
            # 4. Lift back to pre-grasp
            self.job_queue.append(pre_grasp_js)
            
            # 5. Move to basket (above release position)
            rel_x = basket_pos[0] + 0.0
            rel_y = basket_pos[1] - 0.035
            rel_z = basket_pos[2] + 0.185
            
            self.get_logger().info(f'  Step 5: Computing IK for release position...')
            release_js = self.compute_ik(self.joint_state, rel_x, rel_y, rel_z)
            if release_js is None:
                self.get_logger().error(f"IK failed for release pose of {box_name}")
                continue
            self.job_queue.append(release_js)
            
            # 6. Release gripper
            self.job_queue.append('toggle_grip')
            
            # Execute the job queue for this box
            self.execute_jobs()
            
            self.get_logger().info(f'✓ {box_name} placed in {box_color} basket!')
        
        self.get_logger().info('\n🎉 ALL BOXES SORTED! Demo complete.')

    def execute_jobs(self):
        """Execute jobs from the queue (JointState or 'toggle_grip')."""
        if not self.job_queue:
            return
        
        next_job = self.job_queue.pop(0)
        
        if isinstance(next_job, JointState):
            # Plan and execute trajectory
            traj = self.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return
            
            self.get_logger().info("Planned to position, executing...")
            self._execute_joint_trajectory(traj.joint_trajectory)
            
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        """Toggle gripper and proceed to next job."""
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            return
        
        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        result = future.result()
        if result is not None and result.success:
            self.get_logger().info('✓ Gripper toggled')
            # Publish gripper event for visualization
            # Determine if this is grasp or release based on queue state
            if len(self.job_queue) == 0 or (len(self.job_queue) > 0 and not isinstance(self.job_queue[0], str)):
                # This was a release (grasp happens earlier in sequence)
                if self.current_box_name:
                    box_color = next((b["color"] for b in self.boxes if b["name"] == self.current_box_name), "unknown")
                    self.publish_gripper_event(f'release:{self.current_box_name}:{box_color}_basket')
            else:
                # This was a grasp
                if self.current_box_name:
                    self.publish_gripper_event(f'grasp:{self.current_box_name}')
        else:
            self.get_logger().warn('Gripper toggle may have failed')
        
        self.execute_jobs()  # Proceed to next job

    def _execute_joint_trajectory(self, joint_traj):
        """Execute joint trajectory via action client."""
        self.get_logger().info('Waiting for controller action server...')
        self._action_client.wait_for_server()
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        
        self.get_logger().info('Sending trajectory to controller...')
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        """Callback when goal is sent."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected')
            return
        
        self.get_logger().info('Trajectory executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        """Callback when trajectory execution completes."""
        try:
            result = future.result().result
            self.get_logger().info('✓ Trajectory execution complete')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Trajectory execution failed: {e}')

    def compute_ik(self, current_joint_state, x, y, z,
                   qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        """
        Compute an IK solution for the UR5e given a target pose.
        Based on the working IKPlanner implementation.
        """
        # Build the target pose
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        # Build IK request
        ik_req = GetPositionIK.Request()
        ik_req.ik_request = PositionIKRequest()
        
        # MoveIt group name for UR5e arm
        ik_req.ik_request.group_name = 'ur_manipulator'
        
        # Name of the end-effector link
        ik_req.ik_request.ik_link_name = 'tool0'
        
        # Seed state for IK
        ik_req.ik_request.robot_state.joint_state = current_joint_state
        
        # Target pose
        ik_req.ik_request.pose_stamped = pose
        
        # Collision checking and timeout
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout = Duration(sec=2)
        
        # Also set top-level robot_state
        ik_req.robot_state.joint_state = current_joint_state

        # Call service
        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service call failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, error code: {result.error_code.val}')
            return None

        self.get_logger().info('✓ IK solution found')
        return result.solution.joint_state

    def plan_to_joints(self, target_joint_state):
        """
        Plan motion given a desired joint configuration.
        Based on the working IKPlanner implementation.
        """
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.planner_id = "RRTConnectkConfigDefault"

        # Build goal constraints
        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )
        req.motion_plan_request.goal_constraints.append(goal_constraints)

        # Call planning service
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Planning service call failed.')
            return None

        result = future.result()
        if result.motion_plan_response.error_code.val != 1:
            self.get_logger().error(f'Planning failed, error code: {result.motion_plan_response.error_code.val}')
            return None

        self.get_logger().info('✓ Motion plan computed successfully')
        return result.motion_plan_response.trajectory

    def publish_gripper_event(self, event):
        """Publish gripper event to scene manager."""
        msg = String()
        msg.data = event
        self.gripper_event_pub.publish(msg)
        self.get_logger().info(f'  → Gripper event: {event}')


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
