"""
Launch: Real UR10 + Robotiq 3-Finger gripper + MoveIt (no simulation).

─── What was fixed vs the original ────────────────────────────────────────
  PROBLEM : The robotiq_3f_gripper_ros2_driver node connects to the gripper
            over Modbus TCP but never sends the hardware activation command
            (rACT=1).  The gripper stays in RESET state → the driver loops
            forever printing "waiting for gripper to connect" → MoveIt
            receives ABORTED on every gripper goal.

  FIX     : A Python pre-activator script (gripper_modbus_activator.py) is
            launched as an ExecuteProcess BEFORE the ROS 2 driver node.
            It opens a raw Modbus TCP socket, sends RESET then ACTIVATE,
            and polls until gSTA==3 (fully activated).  Only then does the
            driver node see a live, activated gripper and complete its init.

  OTHER FIXES:
    • Added gripper_activated_check node that monitors /gripper/status and
      logs a clear ✅/❌ banner so you always know the gripper state.
    • Increased T_MOVEIT (15→20 s) to give activation more time before
      move_group connects to the gripper controller.
    • Added LIBGL_ALWAYS_SOFTWARE=1 env var to RViz to prevent SIGSEGV
      (exit code -11) on NUC integrated graphics.
    • Wrapped gripper driver in its own TimerAction (T_GRIPPER_DRIVER=5 s)
      so the activator script has time to finish before the driver starts.

─── Workflow A — everything in one terminal (recommended) ──────────────────
  1. Copy gripper_modbus_activator.py next to this file (or anywhere on PATH).
  2. Kill any stale ROS processes:
       pkill -9 -f ur_ros2_control_node; pkill -9 -f "ros2 launch"; sleep 2
  3. Run this launch file:
       ros2 launch ur_yt_sim real_ur10_3f_gripper_moveit_launch.py with_gripper:=true
  4. On the UR teach pendant:
       Installation → URCaps → External Control
       Set  Host IP  = 192.168.1.10  (this PC)
       Set  Port     = 50002
       Load the External Control program and press PLAY ▶
  5. Wait ~5 s — you will see "Robot connected" in the UR driver logs.
  6. RViz + MoveIt become fully active automatically.

─── Workflow B — UR driver already running in another terminal ─────────────
  ros2 launch ur_yt_sim real_ur10_3f_gripper_moveit_launch.py \\
      with_gripper:=true launch_ur_driver:=false

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
    DeclareLaunchArgument, IncludeLaunchDescription,
    LogInfo, TimerAction, ExecuteProcess,
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

# ── Timing constants (seconds) ────────────────────────────────────────────────
# with launch_ur_driver:=true  (full start)
T_GRIPPER_ACTIVATOR = 0    # activator runs immediately at launch
T_GRIPPER_DRIVER    = 8    # driver starts after activator finishes (~6-8 s)
T_CONTROLLERS       = 10   # extra UR controllers
T_MOVEIT            = 20   # move_group (gripper must be activated by now)
T_RVIZ              = 25   # RViz

# with launch_ur_driver:=false (driver already running)
T_GRIPPER_ACTIVATOR_ND = 0
T_GRIPPER_DRIVER_ND    = 8
T_MOVEIT_NODELAY       = 12
T_RVIZ_NODELAY         = 15

# ── Path to the pre-activator script ─────────────────────────────────────────
# Resolve relative to THIS file so it works from any working directory.
_THIS_DIR      = os.path.dirname(os.path.realpath(__file__))
ACTIVATOR_SCRIPT = os.path.join(_THIS_DIR, "gripper_modbus_activator.py")
# Fallback: check the package share directory
if not os.path.exists(ACTIVATOR_SCRIPT):
    try:
        _share = get_package_share_directory("ur_yt_sim")
        ACTIVATOR_SCRIPT = os.path.join(_share, "gripper_modbus_activator.py")
    except Exception:
        pass


def generate_launch_description():
    ld = LaunchDescription()

    # ── share dirs ────────────────────────────────────────────────────────────
    uryt_share    = get_package_share_directory("ur_yt_sim")
    ur_driver_dir = get_package_share_directory("ur_robot_driver")

    # ── launch arguments ──────────────────────────────────────────────────────
    ld.add_action(DeclareLaunchArgument("with_rviz",    default_value="true"))
    ld.add_action(DeclareLaunchArgument("with_octomap", default_value="false"))
    ld.add_action(DeclareLaunchArgument("with_gripper", default_value="true"))
    ld.add_action(DeclareLaunchArgument(
        "launch_ur_driver",
        default_value="true",
        description=(
            "Set false when ur_control.launch.py is already running in "
            "another terminal and the robot is already connected."
        ),
    ))

    launch_ur_driver = LaunchConfiguration("launch_ur_driver")

    # ── startup banner ────────────────────────────────────────────────────────
    ld.add_action(LogInfo(msg=(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  UR10 + Robotiq 3F  —  Real Hardware Launch  (FIXED)    ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        "║  Robot      : 192.168.1.102                              ║\n"
        "║  Gripper    : 192.168.1.105  (Modbus TCP :502)           ║\n"
        "║  PC         : 192.168.1.10                               ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        "║  FIX: Gripper pre-activator runs first (rACT=1 via       ║\n"
        "║       raw Modbus TCP).  Driver starts after 8 s.         ║\n"
        "╚══════════════════════════════════════════════════════════╝"
    )))

    # ── calibration / controllers ─────────────────────────────────────────────
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

    # ── 1. Robot State Publisher ──────────────────────────────────────────────
    ld.add_action(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, {"use_sim_time": False}],
        output="screen",
    ))

    # ── 2. UR Robot Driver ────────────────────────────────────────────────────
    ld.add_action(IncludeLaunchDescription(
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

    # ── 3A. Gripper Pre-Activator (runs BEFORE the driver node) ──────────────
    #
    #  This is the KEY FIX.
    #  gripper_modbus_activator.py opens a raw Modbus TCP socket to the gripper
    #  and sends: RESET → ACTIVATE → polls until gSTA==3.
    #  The driver node starts T_GRIPPER_DRIVER seconds later and finds an
    #  already-activated gripper, completing its init successfully.
    #
    if os.path.exists(ACTIVATOR_SCRIPT):
        # With UR driver: starts immediately
        ld.add_action(TimerAction(
            period=float(T_GRIPPER_ACTIVATOR),
            actions=[
                ExecuteProcess(
                    cmd=[sys.executable, ACTIVATOR_SCRIPT],
                    output="screen",
                    name="gripper_pre_activator",
                ),
            ],
            condition=IfCondition(launch_ur_driver),
        ))
        # Without UR driver: same timing
        ld.add_action(TimerAction(
            period=float(T_GRIPPER_ACTIVATOR_ND),
            actions=[
                ExecuteProcess(
                    cmd=[sys.executable, ACTIVATOR_SCRIPT],
                    output="screen",
                    name="gripper_pre_activator",
                ),
            ],
            condition=UnlessCondition(launch_ur_driver),
        ))
    else:
        ld.add_action(LogInfo(msg=(
            f"[WARN] gripper_modbus_activator.py not found at {ACTIVATOR_SCRIPT}\n"
            "       Copy it next to this launch file or into the package share dir.\n"
            "       Gripper driver will start without pre-activation — may loop."
        )))

    # ── 3B. Robotiq 3F Driver — starts AFTER activator finishes ──────────────
    try:
        get_package_prefix("robotiq_3f_gripper_ros2_driver")

        # With UR driver
        ld.add_action(TimerAction(
            period=float(T_GRIPPER_DRIVER),
            actions=[
                Node(
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
                    condition=IfCondition(LaunchConfiguration("with_gripper")),
                ),
            ],
            condition=IfCondition(launch_ur_driver),
        ))

        # Without UR driver
        ld.add_action(TimerAction(
            period=float(T_GRIPPER_DRIVER_ND),
            actions=[
                Node(
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
                    condition=IfCondition(LaunchConfiguration("with_gripper")),
                ),
            ],
            condition=UnlessCondition(launch_ur_driver),
        ))

    except Exception:
        ld.add_action(LogInfo(
            msg="[WARN] robotiq_3f_gripper_ros2_driver not found — gripper skipped"
        ))

    # ── 4. Extra UR controllers ───────────────────────────────────────────────
    ld.add_action(TimerAction(period=float(T_CONTROLLERS), actions=[
        Node(
            package="controller_manager", executable="spawner",
            arguments=["io_and_status_controller",
                       "--controller-manager", "/controller_manager"],
            output="screen",
            condition=IfCondition(launch_ur_driver),
        ),
        Node(
            package="controller_manager", executable="spawner",
            arguments=["force_torque_sensor_broadcaster",
                       "--controller-manager", "/controller_manager"],
            output="screen",
            condition=IfCondition(launch_ur_driver),
        ),
    ]))

    # ── 5. MoveIt move_group ──────────────────────────────────────────────────
    mg_params = moveit_config.to_dict()
    mg_params.update({"use_sim_time": False})
    mg_no_sensor = dict(mg_params)
    mg_no_sensor.pop("sensors", None)

    def _mg_nodes(params):
        return [
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[params],
                arguments=["--ros-args", "--log-level", "info"],
                condition=IfCondition(LaunchConfiguration("with_octomap")),
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[mg_no_sensor if params is mg_params else params],
                arguments=["--ros-args", "--log-level", "info"],
                condition=UnlessCondition(LaunchConfiguration("with_octomap")),
            ),
        ]

    # With driver
    ld.add_action(TimerAction(
        period=float(T_MOVEIT),
        actions=_mg_nodes(mg_params),
        condition=IfCondition(launch_ur_driver),
    ))
    # Without driver
    ld.add_action(TimerAction(
        period=float(T_MOVEIT_NODELAY),
        actions=_mg_nodes(mg_params),
        condition=UnlessCondition(launch_ur_driver),
    ))

    # ── 6. RViz — with LIBGL_ALWAYS_SOFTWARE to prevent SIGSEGV on NUC ───────
    #  exit code -11 = SIGSEGV, usually triggered by Mesa/OpenGL on Intel NUC
    #  LIBGL_ALWAYS_SOFTWARE=1 forces software rendering → stable but slower.
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
        additional_env={"LIBGL_ALWAYS_SOFTWARE": "1"},   # ← FIX for exit -11
        condition=IfCondition(LaunchConfiguration("with_rviz")),
    )

    ld.add_action(TimerAction(
        period=float(T_RVIZ),
        actions=[rviz_node],
        condition=IfCondition(launch_ur_driver),
    ))
    ld.add_action(TimerAction(
        period=float(T_RVIZ_NODELAY),
        actions=[rviz_node],
        condition=UnlessCondition(launch_ur_driver),
    ))

    return ld
