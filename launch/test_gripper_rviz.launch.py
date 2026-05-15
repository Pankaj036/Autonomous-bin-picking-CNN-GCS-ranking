"""
Test launch: UR10 + Robotiq 3F gripper with FAKE hardware + MoveIt + RViz.
No real robot or gripper needed — everything runs in simulation via mock_components.

Usage:
  ros2 launch ur_yt_sim test_gripper_rviz.launch.py

What runs:
  1. robot_state_publisher   — publishes TF from fake joint states
  2. controller_manager      — fake hardware (mock_components) for arm + gripper
  3. joint_state_broadcaster — publishes /joint_states from mock hardware
  4. joint_trajectory_controller  — for UR10 arm
  5. gripper_action_controller    — GripperCommand action for finger_1_joint_1
  6. MoveIt move_group        — planning for ur10_manipulator + robotiq_3f_gripper
  7. RViz                     — with MoveIt MotionPlanning panel

In RViz:
  - Select planning group  "robotiq_3f_gripper"
  - Set Goal State → "open" or "close" from the dropdown
  - Click Plan → Execute  →  fingers animate in RViz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ld = LaunchDescription()

    uryt_share = get_package_share_directory("ur_yt_sim")

    # ── controller yaml for fake hardware ─────────────────────────────
    controllers_file = os.path.join(uryt_share, "config", "ur10_controllers_fake.yaml")

    # ── MoveIt config  (use_fake_hardware=true → mock_components for arm+gripper)
    moveit_config = (
        MoveItConfigsBuilder(
            "custom_robot",
            package_name="ur10_3f_gripper_moveit_config",
        )
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings={
                "ur_type":            "ur10",
                "sim_gazebo":         "false",
                "sim_ignition":       "false",
                "use_fake_hardware":  "true",
                "mock_sensor_commands": "false",
                "simulation_controllers": controllers_file,
                "initial_positions_file": os.path.join(
                    uryt_share, "config", "initial_positions.yaml"
                ),
            },
        )
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path="config/moveit_controllers_fake.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True,
        )
        .to_moveit_configs()
    )

    # ── 1. robot_state_publisher ──────────────────────────────────────
    ld.add_action(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, {"use_sim_time": False}],
        output="screen",
    ))

    # ── 2. ros2_control controller_manager (fake hardware) ───────────
    ld.add_action(Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            controllers_file,
            {"use_sim_time": False},
        ],
        output="screen",
    ))

    # ── 3. Spawn controllers ──────────────────────────────────────────
    ld.add_action(TimerAction(period=2.0, actions=[
        Node(
            package="controller_manager", executable="spawner",
            arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
            output="screen",
        ),
    ]))
    ld.add_action(TimerAction(period=3.0, actions=[
        Node(
            package="controller_manager", executable="spawner",
            arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
            output="screen",
        ),
        Node(
            package="controller_manager", executable="spawner",
            arguments=["gripper_action_controller", "-c", "/controller_manager"],
            output="screen",
        ),
    ]))

    # ── 4. MoveIt move_group ──────────────────────────────────────────
    mg_params = {**moveit_config.to_dict(), "use_sim_time": False}
    mg_params.pop("sensors", None)

    ld.add_action(TimerAction(period=4.0, actions=[
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[mg_params],
            arguments=["--ros-args", "--log-level", "info"],
        ),
    ]))

    # ── 5. RViz ───────────────────────────────────────────────────────
    rviz_cfg = os.path.join(
        get_package_share_directory("ur10_3f_gripper_moveit_config"),
        "config", "moveit.rviz",
    )
    ld.add_action(TimerAction(period=5.0, actions=[
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_cfg],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.planning_pipelines,
                moveit_config.robot_description_kinematics,
                {"use_sim_time": False},
            ],
        ),
    ]))

    return ld
