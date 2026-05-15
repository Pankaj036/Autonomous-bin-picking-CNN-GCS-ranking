"""
Launch: Real UR10 + Robotiq FT 300-S + Robotiq 3F gripper + OAK-D Cameras + MoveIt

Hardware stack  (tool0 → outward)
───────────────────────────────────
  tool0
    └── Robotiq FT 300-S  (F-0125)         /ft_sensor/wrench
          └── Robotiq 3F Gripper
          └── 15 cm camera bracket
                └── OAK-D Pro Wide (wrist camera)

Camera setup
─────────────
  env_camera   : OAK-D Pro AF  (USB DeviceId 14442C1041A6D1D200, port 3.5)
                 Fixed stand viewing the robot workspace.
                 Topics → /env_camera/rgb/*, /env_camera/stereo/*, /env_camera/points

  wrist_camera : OAK-D Pro Wide  (USB DeviceId 14442C10715AD4D200, port 3.6)
                 Mounted on 15 cm bracket at FT300 output face.
                 Topics → /wrist_camera/rgb/*, /wrist_camera/stereo/*, /wrist_camera/points

Camera TF frames are defined in urdf/oak_d_cameras.xacro and published by RSP.

─── Recommended: everything in one terminal ──────────────────────────────────
  ros2 launch ur_yt_sim real_ur10_oakd_gripper_moveit.launch.py with_gripper:=true

  On the teach pendant:
    Installation → URCaps → External Control
    Host IP = 192.168.1.10   Port = 50002
    Load program → Press PLAY ▶

─── UR driver already running in another terminal ────────────────────────────
  ros2 launch ur_yt_sim real_ur10_oakd_gripper_moveit.launch.py \\
      with_gripper:=true launch_ur_driver:=false

─── Cameras only (skip MoveIt) ──────────────────────────────────────────────
  ros2 launch ur_yt_sim real_ur10_oakd_gripper_moveit.launch.py \\
      with_cameras:=true launch_ur_driver:=false with_rviz:=false

─── With MoveIt octomap (3-D obstacle avoidance from env_camera) ─────────────
  ros2 launch ur_yt_sim real_ur10_oakd_gripper_moveit.launch.py with_octomap:=true

Hardware IPs:
  UR10  robot  : 192.168.1.102
  Robotiq 3F   : 192.168.1.105  (Modbus TCP port 502)
  PC           : 192.168.1.10
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
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode, ParameterFile
from moveit_configs_utils import MoveItConfigsBuilder


# ── Fixed hardware addresses ──────────────────────────────────────────────────
ROBOT_IP   = "192.168.1.102"
GRIPPER_IP = "192.168.1.105"
PC_IP      = "192.168.1.10"

# ── Timing: launch_ur_driver:=true  (full cold start) ────────────────────────
T_GRIPPER_ACTIVATOR = 0    # pre-activator fires immediately
T_GRIPPER_DRIVER    = 8    # gripper driver starts after activator (~6-8 s)
T_CONTROLLERS       = 10   # io_and_status + force_torque spawners
T_MOVEIT            = 20   # move_group (gripper activated, robot connected)
T_RVIZ              = 30   # RViz — extra delay so move_group is fully ready

# ── Timing: launch_ur_driver:=false  (driver already running) ─────────────────
T_GRIPPER_ACTIVATOR_ND = 0
T_GRIPPER_DRIVER_ND    = 8
T_MOVEIT_SHORT         = 12
T_RVIZ_SHORT           = 22  # +10 s past move_group start so MoveGroup plugin connects cleanly

# ── Gripper pre-activator script path ────────────────────────────────────────
_THIS_DIR        = os.path.dirname(os.path.realpath(__file__))
ACTIVATOR_SCRIPT = os.path.join(_THIS_DIR, "gripper_modbus_activator.py")
if not os.path.exists(ACTIVATOR_SCRIPT):
    try:
        _share = get_package_share_directory("ur_yt_sim")
        ACTIVATOR_SCRIPT = os.path.join(_share, "launch", "gripper_modbus_activator.py")
    except Exception:
        pass


def launch_setup(context, *args, **kwargs):

    launch_ur_driver  = LaunchConfiguration("launch_ur_driver")
    with_rviz         = LaunchConfiguration("with_rviz")
    with_octomap      = LaunchConfiguration("with_octomap")
    with_gripper      = LaunchConfiguration("with_gripper")
    with_cameras      = LaunchConfiguration("with_cameras")

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
                "ur_type":            "ur10",
                "sim_gazebo":         "false",
                "sim_ignition":       "false",
                "use_fake_hardware":  "false",
                "robot_ip":           ROBOT_IP,
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
    mg_no_sensor.pop("sensors", None)   # no octomap when with_octomap:=false

    # Prevent move_group from publishing /robot_description (would re-trigger UR
    # driver hardware init).  Also disable the planning_scene_monitor's
    # robot_description topic subscription so move_group keeps the full UR10+3F
    # model it received from the node parameter — it won't reload from the topic
    # if the UR driver's RSP later resends the UR-only model.
    if launch_ur_driver.perform(context) == "false":
        mg_params["publish_robot_description"]          = False
        mg_params["monitor_robot_description"]          = False
        mg_no_sensor["publish_robot_description"]       = False
        mg_no_sensor["monitor_robot_description"]       = False

    actions = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Robot State Publisher
    #    Publishes TF for arm + gripper + OAK-D camera frames.
    #
    #    Always publishes to /robot_description.  When launch_ur_driver:=false
    #    the UR driver's RSP already owns that topic (UR-only model).  Our RSP
    #    starts a few seconds later and overwrites it with the full UR10+3F
    #    model.  move_group (starting at T_MOVEIT_SHORT=12 s) therefore always
    #    sees the complete description.
    #
    #    Side effect: the UR driver may briefly re-sync its hardware interface
    #    when it sees the new /robot_description.  On a clean start (PLAY
    #    already pressed) this is harmless — press PLAY once more on the pendant
    #    if the program stops.
    # ══════════════════════════════════════════════════════════════════════════
    actions.append(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, {"use_sim_time": False}],
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
    #    Fires first so the gripper driver finds an already-active gripper.
    # ══════════════════════════════════════════════════════════════════════════
    if os.path.exists(ACTIVATOR_SCRIPT):
        # Single activator fires once regardless of whether the UR driver is
        # also being launched in this session.  Both T values are 0 so there
        # is no timing difference; the old duplicate-block code ran the
        # activator twice every launch, causing the gripper to cycle twice.
        actions.append(TimerAction(
            period=float(T_GRIPPER_ACTIVATOR),
            actions=[ExecuteProcess(
                cmd=[sys.executable, ACTIVATOR_SCRIPT],
                output="screen",
            )],
            condition=IfCondition(with_gripper),
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Robotiq 3F Gripper Driver
    # ══════════════════════════════════════════════════════════════════════════
    try:
        get_package_prefix("robotiq_3f_gripper_ros2_driver")
        gripper_node = Node(
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
        # with driver: wait for activator to finish
        actions.append(TimerAction(
            period=float(T_GRIPPER_DRIVER),
            actions=[gripper_node],
            condition=IfCondition(launch_ur_driver),
        ))
        # without driver: also wait for activator
        actions.append(TimerAction(
            period=float(T_GRIPPER_DRIVER_ND),
            actions=[gripper_node],
            condition=UnlessCondition(launch_ur_driver),
        ))
    except Exception:
        actions.append(LogInfo(
            msg="[WARN] robotiq_3f_gripper_ros2_driver not found — gripper node skipped"
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Extra UR controllers (io_and_status, force_torque)
    #    Only spawned when we also launch the UR driver (cold start).
    #    When launch_ur_driver:=false these are already active — re-spawning
    #    them crashes FreedriveModeController::on_cleanup in ur_ros2_control_node.
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
    # 6. OAK-D Cameras  (via depthai camera_as_part_of_a_robot.launch.py)
    #
    #   wrist_camera → OAK-D Pro Wide  (MxID 14442C10715AD4D200)
    #   env_camera removed — only one physical camera is connected.
    #
    #   TF frames come from the main robot_state_publisher (URDF oak_d_cameras.xacro).
    # ══════════════════════════════════════════════════════════════════════════

    # ── 6b. Wrist camera ──────────────────────────────────────────────────────
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(depthai_dir, "launch", "camera_as_part_of_a_robot.launch.py")
        ),
        launch_arguments={
            "name":                       "wrist_camera",
            "camera_model":               "OAK-D-PRO-W",
            "params_file":                os.path.join(uryt_share, "config", "oak_d_pro_wide_wrist.yaml"),
            "publish_tf_from_calibration": "false",
        }.items(),
        condition=IfCondition(with_cameras),
    ))

    # NOTE: Robotiq FT 300-S is physically mounted but NOT connected
    # electrically. TF frames (ft300_link / ft300_sensor_frame /
    # ft300_output_link) are published by robot_state_publisher from the URDF.
    # No driver node is launched.

    # ══════════════════════════════════════════════════════════════════════════
    # 7. MoveIt move_group
    #    move_group reads robot_description from the node parameter (full
    #    UR10+3F model injected by MoveItConfigsBuilder).  It also monitors
    #    /robot_description for live updates; by the time move_group starts our
    #    RSP has already published the full model there, so no remapping needed.
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
    # 8. Bin-Picking Pipeline Node
    #    Starts after MoveIt (T_MOVEIT + 5 s) so move_group is ready.
    # ══════════════════════════════════════════════════════════════════════════
    T_PIPELINE = T_MOVEIT + 5
    T_PIPELINE_ND = T_MOVEIT_SHORT + 5
    actions.append(TimerAction(period=float(T_PIPELINE), actions=[
        Node(
            package="ur_yt_sim",
            executable="bin_picking_pipeline_node",
            name="bin_picking_pipeline",
            output="screen",
            parameters=[{
                "device":      "cpu",
                "vel_scale":   0.30,
                "accel_scale": 0.30,
            }],
        ),
    ], condition=IfCondition(launch_ur_driver)))
    actions.append(TimerAction(period=float(T_PIPELINE_ND), actions=[
        Node(
            package="ur_yt_sim",
            executable="bin_picking_pipeline_node",
            name="bin_picking_pipeline",
            output="screen",
            parameters=[{
                "device":      "cpu",
                "vel_scale":   0.30,
                "accel_scale": 0.30,
            }],
        ),
    ], condition=UnlessCondition(launch_ur_driver)))

    # ══════════════════════════════════════════════════════════════════════════
    # 9. RViz
    # ══════════════════════════════════════════════════════════════════════════
    rviz_cfg = os.path.join(
        get_package_share_directory("ur10_3f_gripper_moveit_config"),
        "config", "moveit.rviz",
    )
    rviz_node = Node(
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
        additional_env={"QT_QPA_PLATFORM": "xcb",
                        "LIBGL_ALWAYS_SOFTWARE": "1"},
        condition=IfCondition(with_rviz),
        on_exit=[LogInfo(msg="[INFO] RViz closed or crashed — pipeline continues running.")],
    )

    actions.append(TimerAction(
        period=float(T_RVIZ),
        actions=[rviz_node],
        condition=IfCondition(launch_ur_driver),
    ))
    actions.append(TimerAction(
        period=float(T_RVIZ_SHORT),
        actions=[rviz_node],
        condition=UnlessCondition(launch_ur_driver),
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 10. Wrist Camera View  (RGB + Depth, 1×2 window)
    # ══════════════════════════════════════════════════════════════════════════
    actions.append(TimerAction(
        period=float(T_RVIZ_SHORT) + 5.0,
        actions=[ExecuteProcess(
            cmd=["bash", "-c", "ros2 run ur_yt_sim camera_quad_view"],
            output="screen",
        )],
        condition=IfCondition(with_cameras),
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 11. CNN Pipeline Visualizer
    #    ┌──────────────┬──────────────┬──────────────┐
    #    │  Live RGB    │  Seg/Feature │  GCS Map     │
    #    ├──────────────┼──────────────┼──────────────┤
    #    │  Real Depth  │  CNN Depth   │  Final Grasp │
    #    └──────────────┴──────────────┴──────────────┘
    # ══════════════════════════════════════════════════════════════════════════
    actions.append(TimerAction(
        period=float(T_RVIZ_SHORT) + 7.0,
        actions=[ExecuteProcess(
            cmd=["bash", "-c", "ros2 run ur_yt_sim cnn_vis_node"],
            output="screen",
        )],
        condition=IfCondition(with_cameras),
    ))

    return actions


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(LogInfo(msg=(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  UR10 + Robotiq 3F + OAK-D Cameras  —  Real Hardware        ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  Robot      : 192.168.1.102                                  ║\n"
        "║  Gripper    : 192.168.1.105  (Modbus TCP)                   ║\n"
        "║  PC         : 192.168.1.10                                   ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  env_camera  : OAK-D Pro AF   14442C1041A6D1D200            ║\n"
        "║  wrist_camera: OAK-D Pro Wide 14442C10715AD4D200            ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )))

    ld.add_action(DeclareLaunchArgument(
        "launch_ur_driver", default_value="true",
        description="Set false when ur_control.launch.py is already running.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_rviz", default_value="true",
        description="Launch RViz with MoveIt.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_octomap", default_value="false",
        description="Enable MoveIt octomap from /env_camera/points.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_gripper", default_value="true",
        description="Launch Robotiq 3F gripper driver + pre-activator.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "with_cameras", default_value="true",
        description="Launch both OAK-D camera drivers.",
    ))

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
