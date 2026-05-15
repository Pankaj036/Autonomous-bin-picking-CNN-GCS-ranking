"""
bin_picking_full_real.launch.py
================================
ONE-FILE launch for the complete CAS-GCS bin-picking pipeline on real hardware.

Includes:
  1. Robot State Publisher (UR10 + 3F gripper + OAK-D TF frames)
  2. UR10 robot driver  (192.168.1.102)
  3. Robotiq 3F gripper pre-activator + driver  (192.168.1.105)
  4. Extra UR controllers  (io_and_status, force_torque)
  5. OAK-D Pro AF      env_camera   (MxID 14442C1041A6D1D200)
  6. OAK-D Pro Wide    wrist_camera (MxID 14442C10715AD4D200)
  7. MoveIt move_group
  8. RViz
  9. Camera quad-view window
 10. MoveIt workspace collision scene
 11. Bin-picking CAS-GCS pipeline node  (CNN + grasp planning + execution)

Topic remappings for pipeline node:
  /wrist_camera/rgb/image_raw           → /camera/color/image_raw
  /wrist_camera/stereo/image_raw        → /camera/aligned_depth_to_color/image_raw
  /wrist_camera/points                  → /camera/depth_registered/points
  /wrist_camera/rgb/camera_info         → /camera/color/camera_info

Usage:
  # Full cold start (everything):
  ros2 launch ur_yt_sim bin_picking_full_real.launch.py

  # Skip robot driver (already running):
  ros2 launch ur_yt_sim bin_picking_full_real.launch.py launch_ur_driver:=false

  # Without cameras (bench-test CNN only):
  ros2 launch ur_yt_sim bin_picking_full_real.launch.py with_cameras:=false

  # Trigger one grasp cycle after launch:
  ros2 topic pub --once /pipeline/trigger std_msgs/msg/Bool "data: true"

Hardware IPs:
  UR10     : 192.168.1.102
  Robotiq  : 192.168.1.105  (Modbus TCP 502)
  PC       : 192.168.1.10
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# ── Fixed hardware addresses ──────────────────────────────────────────────────
ROBOT_IP   = "192.168.1.102"
GRIPPER_IP = "192.168.1.105"
PC_IP      = "192.168.1.10"

# ── CNN checkpoint ────────────────────────────────────────────────────────────
CNN_CKPT = "/home/user/Bin-Picking-CAS-GCS/CNN_src/checkpoints/model_9.pth"

# ── Startup timing (cold start: launch_ur_driver:=true) ──────────────────────
T_GRIPPER_ACTIVATOR = 0     # pre-activator fires immediately
T_GRIPPER_DRIVER    = 8     # after activator completes (~6-8 s)
T_CONTROLLERS       = 10    # io_and_status + force_torque spawners
T_MOVEIT            = 20    # move_group (gripper activated + robot connected)
T_RVIZ              = 25    # RViz
T_SCENE             = 30    # workspace collision scene
T_PIPELINE          = 35    # bin-picking pipeline node

# ── Startup timing (warm start: launch_ur_driver:=false) ─────────────────────
T_GRIPPER_ACTIVATOR_ND = 0
T_GRIPPER_DRIVER_ND    = 8
T_MOVEIT_SHORT         = 12
T_RVIZ_SHORT           = 15
T_SCENE_SHORT          = 18
T_PIPELINE_SHORT       = 22

# ── Gripper pre-activator ─────────────────────────────────────────────────────
_THIS_DIR        = os.path.dirname(os.path.realpath(__file__))
ACTIVATOR_SCRIPT = os.path.join(_THIS_DIR, "gripper_modbus_activator.py")
if not os.path.exists(ACTIVATOR_SCRIPT):
    try:
        _share = get_package_share_directory("ur_yt_sim")
        ACTIVATOR_SCRIPT = os.path.join(_share, "launch", "gripper_modbus_activator.py")
    except Exception:
        pass


def launch_setup(context, *args, **kwargs):

    launch_ur_driver = LaunchConfiguration("launch_ur_driver")
    with_rviz        = LaunchConfiguration("with_rviz")
    with_octomap     = LaunchConfiguration("with_octomap")
    with_gripper     = LaunchConfiguration("with_gripper")
    with_cameras     = LaunchConfiguration("with_cameras")
    with_pipeline    = LaunchConfiguration("with_pipeline")

    uryt_share    = get_package_share_directory("ur_yt_sim")
    ur_driver_dir = get_package_share_directory("ur_robot_driver")
    depthai_dir   = get_package_share_directory("depthai_ros_driver")

    # ── Calibration / controller config ──────────────────────────────────────
    calib_file = os.path.join(uryt_share, "config", "ur10_calibration.yaml")
    if not os.path.exists(calib_file):
        calib_file = os.path.join(
            get_package_share_directory("ur_description"),
            "config", "ur10", "default_kinematics.yaml",
        )

    joint_controllers_file = os.path.join(
        uryt_share, "config", "ur10_controllers_real.yaml"
    )

    # ── MoveIt config ─────────────────────────────────────────────────────────
    moveit_config = (
        MoveItConfigsBuilder(
            "custom_robot",
            package_name="ur10_3f_gripper_moveit_config",
        )
        .robot_description(
            file_path="config/ur.urdf.xacro",
            mappings={
                "ur_type":                "ur10",
                "sim_gazebo":             "false",
                "sim_ignition":           "false",
                "use_fake_hardware":      "false",
                "robot_ip":               ROBOT_IP,
                "simulation_controllers": joint_controllers_file,
                "initial_positions_file": os.path.join(
                    uryt_share, "config", "initial_positions.yaml"
                ),
            },
        )
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path="config/moveit_controllers_real.yaml")
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

    mg_params    = moveit_config.to_dict()
    mg_params.update({"use_sim_time": False})
    mg_no_sensor = dict(mg_params)
    mg_no_sensor.pop("sensors", None)

    if launch_ur_driver.perform(context) == "false":
        mg_params["publish_robot_description"]    = False
        mg_no_sensor["publish_robot_description"] = False

    actions = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Robot State Publisher
    # ══════════════════════════════════════════════════════════════════════════
    _rsp_remaps = (
        []
        if launch_ur_driver.perform(context) == "true"
        else [("/robot_description", "/ur_yt_sim/robot_description")]
    )
    actions.append(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, {"use_sim_time": False}],
        remappings=_rsp_remaps,
        output="screen",
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. UR10 Robot Driver
    # ══════════════════════════════════════════════════════════════════════════
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_driver_dir, "launch", "ur_control.launch.py")
        ),
        launch_arguments={
            "ur_type":                    "ur10",
            "robot_ip":                   ROBOT_IP,
            "reverse_ip":                 PC_IP,
            "reverse_port":               "50002",
            "use_fake_hardware":          "false",
            "launch_rviz":                "false",
            "controller_spawner_timeout": "60",
            "kinematics_params":          calib_file,
        }.items(),
        condition=IfCondition(launch_ur_driver),
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Robotiq 3F Gripper Pre-Activator  (raw Modbus: RESET → ACTIVATE)
    # ══════════════════════════════════════════════════════════════════════════
    if os.path.exists(ACTIVATOR_SCRIPT):
        actions.append(TimerAction(
            period=float(T_GRIPPER_ACTIVATOR),
            actions=[ExecuteProcess(
                cmd=[sys.executable, ACTIVATOR_SCRIPT],
                output="screen",
            )],
            condition=IfCondition(launch_ur_driver),
        ))
        actions.append(TimerAction(
            period=float(T_GRIPPER_ACTIVATOR_ND),
            actions=[ExecuteProcess(
                cmd=[sys.executable, ACTIVATOR_SCRIPT],
                output="screen",
            )],
            condition=UnlessCondition(launch_ur_driver),
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Robotiq 3F Gripper Driver
    # ══════════════════════════════════════════════════════════════════════════
    try:
        get_package_prefix("robotiq_3f_gripper_ros2_driver")
        def _gripper_node():
            return Node(
                package="robotiq_3f_gripper_ros2_driver",
                executable="robotiq_3f_gripper_node",
                name="robotiq_3f_gripper",
                output="screen",
                parameters=[{
                    "gripper_ip":   GRIPPER_IP,
                    "gripper_port": 502,
                    "speed":        150,
                    "force":        100,
                }],
                condition=IfCondition(with_gripper),
            )
        actions.append(TimerAction(
            period=float(T_GRIPPER_DRIVER),
            actions=[_gripper_node()],
            condition=IfCondition(launch_ur_driver),
        ))
        actions.append(TimerAction(
            period=float(T_GRIPPER_DRIVER_ND),
            actions=[_gripper_node()],
            condition=UnlessCondition(launch_ur_driver),
        ))
    except Exception:
        actions.append(LogInfo(
            msg="[WARN] robotiq_3f_gripper_ros2_driver not found — gripper node skipped"
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Extra UR controllers  (io_and_status, force_torque)
    # ══════════════════════════════════════════════════════════════════════════
    actions.append(TimerAction(period=float(T_CONTROLLERS), actions=[
        Node(package="controller_manager", executable="spawner",
             arguments=["io_and_status_controller", "--controller-manager", "/controller_manager"],
             output="screen", condition=IfCondition(launch_ur_driver)),
        Node(package="controller_manager", executable="spawner",
             arguments=["force_torque_sensor_broadcaster", "--controller-manager", "/controller_manager"],
             output="screen", condition=IfCondition(launch_ur_driver)),
    ]))

    # ══════════════════════════════════════════════════════════════════════════
    # 6. OAK-D Cameras  (via depthai camera_as_part_of_a_robot)
    #
    #   env_camera   → OAK-D Pro AF    (MxID 14442C1041A6D1D200, port 3.5)
    #     Topics: /env_camera/rgb/image_raw, /env_camera/stereo/image_raw,
    #             /env_camera/points
    #
    #   wrist_camera → OAK-D Pro Wide  (MxID 14442C10715AD4D200, port 3.6)
    #     Topics: /wrist_camera/rgb/image_raw, /wrist_camera/stereo/image_raw,
    #             /wrist_camera/points
    # ══════════════════════════════════════════════════════════════════════════

    # ── 6a. Environment camera ────────────────────────────────────────────────
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(depthai_dir, "launch", "camera_as_part_of_a_robot.launch.py")
        ),
        launch_arguments={
            "name":                        "env_camera",
            "camera_model":                "OAK-D-PRO",
            "params_file":                 os.path.join(uryt_share, "config", "oak_d_pro_env.yaml"),
            "publish_tf_from_calibration": "false",
        }.items(),
        condition=IfCondition(with_cameras),
    ))

    # ── 6b. Wrist camera ──────────────────────────────────────────────────────
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(depthai_dir, "launch", "camera_as_part_of_a_robot.launch.py")
        ),
        launch_arguments={
            "name":                        "wrist_camera",
            "camera_model":                "OAK-D-PRO-W",
            "params_file":                 os.path.join(uryt_share, "config", "oak_d_pro_wide_wrist.yaml"),
            "publish_tf_from_calibration": "false",
        }.items(),
        condition=IfCondition(with_cameras),
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 7. MoveIt move_group
    # ══════════════════════════════════════════════════════════════════════════
    actions.append(TimerAction(period=float(T_MOVEIT), actions=[
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen", parameters=[mg_params],
             arguments=["--ros-args", "--log-level", "info"],
             condition=IfCondition(LaunchConfiguration("with_octomap"))),
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen", parameters=[mg_no_sensor],
             arguments=["--ros-args", "--log-level", "info"],
             condition=UnlessCondition(LaunchConfiguration("with_octomap"))),
    ], condition=IfCondition(launch_ur_driver)))

    actions.append(TimerAction(period=float(T_MOVEIT_SHORT), actions=[
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen", parameters=[mg_params],
             arguments=["--ros-args", "--log-level", "info"],
             condition=IfCondition(LaunchConfiguration("with_octomap"))),
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen", parameters=[mg_no_sensor],
             arguments=["--ros-args", "--log-level", "info"],
             condition=UnlessCondition(LaunchConfiguration("with_octomap"))),
    ], condition=UnlessCondition(launch_ur_driver)))

    # ══════════════════════════════════════════════════════════════════════════
    # 8. RViz
    # ══════════════════════════════════════════════════════════════════════════
    rviz_cfg = os.path.join(
        get_package_share_directory("ur10_3f_gripper_moveit_config"),
        "config", "moveit.rviz",
    )
    def _rviz_node():
        return Node(
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
            additional_env={"QT_QPA_PLATFORM": "xcb", "LIBGL_ALWAYS_SOFTWARE": "1"},
            condition=IfCondition(with_rviz),
        )

    actions.append(TimerAction(
        period=float(T_RVIZ),
        actions=[_rviz_node()],
        condition=IfCondition(launch_ur_driver),
    ))
    actions.append(TimerAction(
        period=float(T_RVIZ_SHORT),
        actions=[_rviz_node()],
        condition=UnlessCondition(launch_ur_driver),
    ))

    # Camera quad-view disabled (window closed by user request)

    # ══════════════════════════════════════════════════════════════════════════
    # 10. MoveIt Workspace Collision Scene
    #     Adds table, pick_bin, place_bin, env_camera as collision objects.
    #     Waits until move_group is ready.
    # ══════════════════════════════════════════════════════════════════════════
    scene_script = os.path.join(
        get_package_share_directory("ur_yt_sim"),
        "scripts", "ur10_workspace_scene.py",
    )
    if not os.path.exists(scene_script):
        scene_script = os.path.join(_THIS_DIR, "..", "scripts", "ur10_workspace_scene.py")

    actions.append(TimerAction(
        period=float(T_SCENE),
        actions=[ExecuteProcess(
            cmd=[sys.executable, scene_script],
            output="screen",
        )],
        condition=IfCondition(launch_ur_driver),
    ))
    actions.append(TimerAction(
        period=float(T_SCENE_SHORT),
        actions=[ExecuteProcess(
            cmd=[sys.executable, scene_script],
            output="screen",
        )],
        condition=UnlessCondition(launch_ur_driver),
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 11. Bin-Picking CAS-GCS Pipeline Node
    #
    #     The pipeline node uses ROS-agnostic topic names:
    #       /camera/color/image_raw              ← wrist RGB
    #       /camera/aligned_depth_to_color/image_raw ← wrist depth
    #       /camera/depth_registered/points      ← wrist point cloud
    #       /camera/color/camera_info            ← wrist camera info
    #       /env_camera/rgb/image_raw            ← env camera (no remap needed)
    #
    #     OAK-D driver publishes on:
    #       /wrist_camera/rgb/image_raw
    #       /wrist_camera/stereo/image_raw       (disparity/depth)
    #       /wrist_camera/points
    #       /wrist_camera/rgb/camera_info
    #
    #     Remappings bridge the two namespaces.
    # ══════════════════════════════════════════════════════════════════════════
    pipeline_script = os.path.join(
        get_package_share_directory("ur_yt_sim"),
        "scripts", "bin_picking_pipeline_node.py",
    )
    if not os.path.exists(pipeline_script):
        pipeline_script = os.path.join(_THIS_DIR, "..", "scripts", "bin_picking_pipeline_node.py")

    def _pipeline_node():
        return Node(
            package="ur_yt_sim",
            executable="bin_picking_pipeline_node",
            name="bin_picking_pipeline",
            output="screen",
            parameters=[{
                # CNN
                "checkpoint":          CNN_CKPT,
                "device":              "cpu",
                "score_thresh":        0.30,
                "has_gcs_branch":      False,
                # workspace
                "place_offset_y":      0.60,
                "datum_z":             0.54,
                "scan_height":         0.35,
                # arm speed
                "vel_scale":           0.10,
                "accel_scale":         0.15,
                "use_sim_time":        False,
                # env camera — OAK-D Pro AF on 90-cm stand in front of robot
                # TF frame published by depthai_ros_driver with name="env_camera"
                "env_cam_tf_frame":    "env_camera_rgb_camera_optical_frame",
                # fallback position (metres, world frame) when TF is unavailable
                "env_cam_x":           1.00,   # ~1 m in front of robot base
                "env_cam_y":           0.00,
                "env_cam_z":           0.90,   # 90 cm stand height
            }],
            remappings=[
                # Wrist camera → pipeline expected topics
                ("/camera/color/image_raw",
                 "/wrist_camera/rgb/image_raw"),
                ("/camera/aligned_depth_to_color/image_raw",
                 "/wrist_camera/stereo/image_raw"),
                ("/camera/depth_registered/points",
                 "/wrist_camera/points"),
                ("/camera/color/camera_info",
                 "/wrist_camera/rgb/camera_info"),
            ],
            condition=IfCondition(with_pipeline),
        )

    actions.append(TimerAction(
        period=float(T_PIPELINE),
        actions=[_pipeline_node()],
        condition=IfCondition(launch_ur_driver),
    ))
    actions.append(TimerAction(
        period=float(T_PIPELINE_SHORT),
        actions=[_pipeline_node()],
        condition=UnlessCondition(launch_ur_driver),
    ))

    return actions


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(LogInfo(msg=(
        "\n"
        "╔═══════════════════════════════════════════════════════════════════╗\n"
        "║   CAS-GCS BIN-PICKING  —  Full Real Hardware Pipeline            ║\n"
        "╠═══════════════════════════════════════════════════════════════════╣\n"
        "║  Robot      : UR10  @ 192.168.1.102                              ║\n"
        "║  Gripper    : Robotiq 3F  @ 192.168.1.105 (Modbus TCP)          ║\n"
        "║  PC         : 192.168.1.10                                        ║\n"
        "╠═══════════════════════════════════════════════════════════════════╣\n"
        "║  env_camera  : OAK-D Pro AF   14442C1041A6D1D200  (port 3.5)   ║\n"
        "║  wrist_camera: OAK-D Pro Wide 14442C10715AD4D200  (port 3.6)   ║\n"
        "╠═══════════════════════════════════════════════════════════════════╣\n"
        "║  CNN         : ResNet50-FPN + BTS depth + GCS head               ║\n"
        "║  Grasp plan  : GDI2 scoring via CAS algorithm                    ║\n"
        "║  DBCC        : depth-based collision check (thresh=0.02 m)       ║\n"
        "║  Place offset: 60 cm in Y                                         ║\n"
        "╠═══════════════════════════════════════════════════════════════════╣\n"
        "║  BEFORE LAUNCH: on teach pendant                                  ║\n"
        "║    Installation → URCaps → External Control                       ║\n"
        "║    Host IP = 192.168.1.10   Port = 50002                         ║\n"
        "║    Load program → Press PLAY ▶                                    ║\n"
        "╠═══════════════════════════════════════════════════════════════════╣\n"
        "║  Trigger grasp cycle:                                             ║\n"
        "║    ros2 topic pub --once /pipeline/trigger std_msgs/msg/Bool      ║\n"
        "║                   \"data: true\"                                    ║\n"
        "╚═══════════════════════════════════════════════════════════════════╝"
    )))

    ld.add_action(DeclareLaunchArgument(
        "launch_ur_driver", default_value="true",
        description="Set false when ur_control.launch.py is already running.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_rviz", default_value="true",
        description="Launch RViz with MoveIt motion planning display.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_octomap", default_value="false",
        description="Enable MoveIt octomap (3-D obstacle avoidance from env_camera/points).",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_gripper", default_value="true",
        description="Launch Robotiq 3F gripper pre-activator + driver.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_cameras", default_value="true",
        description="Launch both OAK-D camera drivers.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_pipeline", default_value="true",
        description="Launch the CAS-GCS bin-picking pipeline node.",
    ))

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
