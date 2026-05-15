"""
sim_visual_servoing.launch.py
------------------------------
Launches Gazebo + MoveIt for the visual-servoing simulation demo.

Uses ur10_camera_gripper_moveit_config (in source tree) with a
simulation-specific MoveIt controllers file that maps to
joint_trajectory_controller instead of scaled_joint_trajectory_controller.

Usage:
  ros2 launch ur_yt_sim sim_visual_servoing.launch.py
  ros2 launch ur_yt_sim sim_visual_servoing.launch.py with_rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ld = LaunchDescription()

    uryt_share    = get_package_share_directory("ur_yt_sim")
    ur_share      = get_package_share_directory("ur_description")
    robotiq_share = get_package_share_directory("robotiq_description")
    gazebo_dir    = get_package_share_directory("gazebo_ros")
    world_file    = os.path.join(uryt_share, "worlds", "world2.world")

    # ── Gazebo environment ───────────────────────────────────────────────────
    ld.add_action(SetEnvironmentVariable(
        name="GAZEBO_RESOURCE_PATH",
        value=":".join(["/usr/share/gazebo-11", uryt_share, robotiq_share, ur_share]),
    ))
    ld.add_action(SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=":".join([
            os.path.join(uryt_share, "models"),
            os.path.join(robotiq_share, "models"),
            os.path.expanduser("~/.gazebo/models"),
        ]),
    ))
    ld.add_action(SetEnvironmentVariable(
        name="GAZEBO_PLUGIN_PATH",
        value=":".join([
            "/opt/ros/humble/lib",
            os.path.normpath(os.path.join(uryt_share, "..", "..", "lib")),
        ]),
    ))

    # ── Args ─────────────────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument("with_rviz",    default_value="true"))
    ld.add_action(DeclareLaunchArgument("with_octomap", default_value="false"))

    # ── MoveIt config ────────────────────────────────────────────────────────
    # Use ur10_camera_gripper_moveit_config (present in source tree) with the
    # simulation controllers file that uses joint_trajectory_controller.
    sim_controllers_file  = os.path.join(uryt_share, "config", "ur10_controllers_gripper.yaml")
    moveit_controllers_sim = os.path.join(uryt_share, "config", "moveit_controllers_sim.yaml")

    moveit_config = (
        MoveItConfigsBuilder(
            "custom_robot",
            package_name="ur10_camera_gripper_moveit_config",
        )
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings={
                "ur_type":           "ur10",
                "sim_gazebo":        "true",
                "sim_ignition":      "false",
                "use_fake_hardware": "false",
                "simulation_controllers": sim_controllers_file,
                "initial_positions_file": os.path.join(
                    uryt_share, "config", "initial_positions.yaml"
                ),
            },
        )
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path=moveit_controllers_sim)
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True,
        )
        .to_moveit_configs()
    )

    # ── Gazebo ───────────────────────────────────────────────────────────────
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_dir, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "gui":          "true",
            "paused":       "true",
            "world":        world_file,
        }.items(),
    ))

    # ── Robot state publisher ────────────────────────────────────────────────
    ld.add_action(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, {"use_sim_time": True}],
        output="screen",
    ))

    # ── Spawn robot ──────────────────────────────────────────────────────────
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "cobot", "-topic", "robot_description",
                   "-x", "0", "-y", "0", "-z", "0"],
        output="screen",
    )
    ld.add_action(TimerAction(period=3.0, actions=[spawn]))

    # ── Controllers (spawned after robot appears) ────────────────────────────
    jsb  = Node(package="controller_manager", executable="spawner",
                arguments=["joint_state_broadcaster",
                           "--controller-manager", "/controller_manager"],
                output="screen")
    arm  = Node(package="controller_manager", executable="spawner",
                arguments=["joint_trajectory_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen")
    grip = Node(package="controller_manager", executable="spawner",
                arguments=["gripper_position_controller",
                           "--controller-manager", "/controller_manager"],
                output="screen")

    ld.add_action(RegisterEventHandler(
        OnProcessStart(target_action=spawn, on_start=[
            TimerAction(period=2.0, actions=[jsb]),
            TimerAction(period=3.5, actions=[arm, grip]),
        ])
    ))

    # ── RViz ─────────────────────────────────────────────────────────────────
    rviz_cfg = os.path.join(
        get_package_share_directory("ur10_camera_gripper_moveit_config"),
        "config", "moveit.rviz",
    )
    ld.add_action(Node(
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
            {"use_sim_time": True},
        ],
        condition=IfCondition(LaunchConfiguration("with_rviz")),
    ))

    # ── MoveGroup ────────────────────────────────────────────────────────────
    mg_params = moveit_config.to_dict()
    mg_params.update({"use_sim_time": True})

    mg_no_sensor = dict(mg_params)
    mg_no_sensor.pop("sensors", None)

    ld.add_action(Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[mg_params],
        arguments=["--ros-args", "--log-level", "info"],
        condition=IfCondition(LaunchConfiguration("with_octomap")),
    ))
    ld.add_action(Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[mg_no_sensor],
        arguments=["--ros-args", "--log-level", "info"],
        condition=UnlessCondition(LaunchConfiguration("with_octomap")),
    ))

    return ld
