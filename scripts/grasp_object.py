#!/usr/bin/env python3
"""
Simple grasp script — UR10 + Robotiq 3F gripper.

Usage (after launching the robot + MoveIt):
  python3 ~/ur_ws/src/ur_yt_sim/scripts/grasp_object.py

What it does:
  1. Open gripper  (via MoveIt gripper planning group)
  2. Move arm to home position
  3. Move arm to pre-grasp pose (above object)
  4. Lower to grasp pose
  5. Close gripper (via MoveIt gripper planning group)
  6. Lift arm up
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

# MoveIt Python API (moveit_commander wraps the C++ MoveGroupInterface)
import moveit_commander
import sys


class GraspNode(Node):

    def __init__(self):
        super().__init__('grasp_node')
        moveit_commander.roscpp_initialize(sys.argv)

        # Arm planning group
        self.arm = moveit_commander.MoveGroupCommander('ur10_manipulator')
        self.arm.set_max_velocity_scaling_factor(0.15)
        self.arm.set_max_acceleration_scaling_factor(0.1)
        self.arm.set_planning_time(5.0)
        self.arm.set_num_planning_attempts(5)

        # Gripper planning group (uses MoveIt → /gripper/gripper_cmd action)
        self.gripper = moveit_commander.MoveGroupCommander('robotiq_3f_gripper')
        self.gripper.set_planning_time(3.0)
        self.gripper.set_num_planning_attempts(5)

        self.get_logger().info('GraspNode ready')

    # ── Gripper (via MoveIt planning group) ───────────────────────────────────

    def open_gripper(self) -> bool:
        """Plan and execute gripper open via MoveIt (finger_1_joint_1 = 0.0 rad)."""
        self.get_logger().info('Opening gripper ...')
        self.gripper.set_named_target('open')
        ok = self.gripper.go(wait=True)
        self.gripper.stop()
        self.gripper.clear_pose_targets()
        return bool(ok)

    def close_gripper(self, position: float = 1.8) -> bool:
        """
        Plan and execute gripper close via MoveIt.

        position (finger_1_joint_1, rad):
          0.0  = fully open
          1.0  = large object
          1.5  = medium object
          1.8  = default (good general grasp)
          2.0  = small object
          2.44 = fully closed
        """
        self.get_logger().info(f'Closing gripper to {position:.2f} rad ...')
        self.gripper.set_joint_value_target({'finger_1_joint_1': position})
        ok = self.gripper.go(wait=True)
        self.gripper.stop()
        self.gripper.clear_pose_targets()
        return bool(ok)

    # ── Arm ───────────────────────────────────────────────────────────────────

    def move_named(self, name: str) -> bool:
        self.get_logger().info(f'Moving arm → "{name}"')
        self.arm.set_named_target(name)
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return bool(ok)

    def move_pose(self, x: float, y: float, z: float,
                  ox: float = 0.0, oy: float = 1.0,
                  oz: float = 0.0, ow: float = 0.0) -> bool:
        pose = Pose()
        pose.position.x    = x
        pose.position.y    = y
        pose.position.z    = z
        pose.orientation.x = ox
        pose.orientation.y = oy
        pose.orientation.z = oz
        pose.orientation.w = ow
        self.get_logger().info(f'Moving arm → ({x:.3f}, {y:.3f}, {z:.3f})')
        self.arm.set_pose_target(pose)
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return bool(ok)

    # ── Grasp sequence ────────────────────────────────────────────────────────

    def run(self):
        # ── Tune these for your setup ─────────────────────────────────────────
        OBJ_X        = 0.40   # object position in robot base frame (metres)
        OBJ_Y        = 0.00
        OBJ_Z        = 0.18   # height of object top surface
        PRE_GRASP_Z  = 0.38   # hover height before descending
        LIFT_Z       = 0.50   # height after grasping
        GRIPPER_CLOSE = 1.8   # rad, 0=open 2.44=fully closed — tune for object
        # ─────────────────────────────────────────────────────────────────────

        # 1. Open gripper
        self.open_gripper()

        # 2. Go to home (safe start position)
        if not self.move_named('home'):
            self.get_logger().error('Cannot reach home — abort')
            return

        # 3. Pre-grasp: hover above object
        if not self.move_pose(OBJ_X, OBJ_Y, PRE_GRASP_Z):
            self.get_logger().error('Cannot reach pre-grasp — abort')
            return

        # 4. Descend to grasp height
        if not self.move_pose(OBJ_X, OBJ_Y, OBJ_Z):
            self.get_logger().error('Cannot descend to grasp — abort')
            return

        # 5. Close gripper
        self.close_gripper(position=GRIPPER_CLOSE)

        # 6. Lift object
        if not self.move_pose(OBJ_X, OBJ_Y, LIFT_Z):
            self.get_logger().error('Cannot lift — object may have shifted')
            return

        self.get_logger().info('=== Grasp complete! ===')

        # 7. Return to home with object
        self.move_named('home')


def main():
    rclpy.init(args=sys.argv)
    node = GraspNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
