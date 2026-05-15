# ============================================================================
#  bin_picking_cas_gcs_real.launch.py
#  Full CAS-GCS Bin-Picking Pipeline — REAL HARDWARE  (ur_ws)
#
#  Hardware:
#    UR10 CB3   : 192.168.1.102  (Polyscope 3.15.8)
#    Env OAK-D  : mxid=14442C1041A6D1D200  (USB 3.5) — overhead
#    Wrist OAK-D: mxid=14442C10715AD4D200  (USB 3.6) — on gripper
#
#  Pipeline stages:
#    ENV_DETECT  → MOVE_TO_BIN → WRIST_CAPTURE → CNN_INFER
#    → GRASP_PLAN → GRASP_EXEC → PLACE → IDLE
#
#  Usage:
#    cd /home/user/ur_ws
#    colcon build --packages-select ur_yt_sim
#    source install/setup.bash
#    ros2 launch ur_yt_sim bin_picking_cas_gcs_real.launch.py
#
#  Trigger one grasp cycle:
#    ros2 topic pub --once /pipeline/trigger std_msgs/msg/Bool "data: true"
# ============================================================================

import os
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, OpaqueFunction,
                             LogInfo, TimerAction, IncludeLaunchDescription,
                             ExecuteProcess)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# ── Paths ─────────────────────────────────────────────────────────────────────
PKG_DIR      = os.path.join(os.path.dirname(__file__), '..')
SCRIPTS_DIR  = os.path.join(PKG_DIR, 'scripts')
CONFIG_DIR   = os.path.join(PKG_DIR, 'config')

# CNN checkpoint
CNN_CHECKPOINT = '/home/user/dataset/final_model.pth'

# ── Real hardware constants ───────────────────────────────────────────────────
UR_ROBOT_IP    = '192.168.1.102'
GRIPPER_IP     = '192.168.1.105'        # Robotiq 3F gripper Modbus TCP (Hilscher gateway)
WRIST_CAM_MXID = '14442C10715AD4D200'   # OAK-D Pro Wide wrist camera

# ── Camera param files (written at launch) ────────────────────────────────────
CAM_PARAM_DIR    = os.path.join(CONFIG_DIR, 'cam_params')
WRIST_CAM_PARAMS = os.path.join(CAM_PARAM_DIR, 'wrist_cam_params.yaml')


def _write_cam_params():
    os.makedirs(CAM_PARAM_DIR, exist_ok=True)
    with open(WRIST_CAM_PARAMS, 'w') as f:
        f.write(f"""wrist_cam:
  ros__parameters:
    camera:
      i_mx_id: "{WRIST_CAM_MXID}"
      i_nn_type: "none"
    rgb:
      i_resolution: "480"
      i_fps: 10.0
      i_width: 640
      i_height: 480
""")


def launch_setup(context, *args, **kwargs):
    _write_cam_params()

    ur_type      = LaunchConfiguration('ur_type')
    launch_rviz  = LaunchConfiguration('launch_rviz')
    place_offset = LaunchConfiguration('place_offset_y')
    moveit_pkg   = LaunchConfiguration('moveit_config_pkg')

    # NOTE: UR driver, robot_state_publisher, and controller spawners are handled
    # by ur_control.launch.py which must be run FIRST in a separate terminal:
    #   ros2 launch ur_robot_driver ur_control.launch.py \
    #     ur_type:=ur10 robot_ip:=192.168.1.102 launch_rviz:=false
    # Then press Play on the teach pendant to connect the robot.
    # IMPORTANT: Kill any previous instance of this launch before restarting.
    #   The UR driver's robot_state_publisher is the sole RSP — we do NOT start
    #   a second one here to avoid /robot_description and /tf conflicts.

    # ── 0. Kill stale gripper/pipeline nodes from previous launch runs ─────────
    # Prevents duplicate robotiq_3f_gripper and bin_picking_pipeline nodes that
    # cause PREEMPTED and "more than one action server" errors.
    kill_stale = ExecuteProcess(
        cmd=['bash', '-c',
             'pkill -f "robotiq_3f_gripper_node" 2>/dev/null; '
             'pkill -f "bin_picking_pipeline" 2>/dev/null; '
             'pkill -f "workspace_scene" 2>/dev/null; '
             'pkill -f "move_group" 2>/dev/null; '
             'sleep 2; true'],
        output='screen',
        name='kill_stale_nodes',
    )

    # ── 1. MoveIt (ur10_camera_moveit_config from ur_ws) ──────────────────
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare(moveit_pkg), '/launch/move_group.launch.py',
        ]),
    )
    moveit_rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', [FindPackageShare(moveit_pkg), '/config/moveit.rviz']],
        output='log',
        # LIBGL_ALWAYS_SOFTWARE=1: force Mesa software rendering to avoid
        # Intel GPU segfault on NUC14 with RViz2/Humble.
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
        condition=__import__('launch.conditions', fromlist=['IfCondition']).IfCondition(launch_rviz),
    )

    # ── 5. Wrist camera (OAK-D USB 3.6) — only camera used ───────────────
    wrist_cam = Node(
        package='depthai_ros_driver', executable='camera_node',
        name='wrist_cam', output='screen',
        parameters=[WRIST_CAM_PARAMS],
        remappings=[
            ('~/rgb/image_raw',    '/camera/color/image_raw'),
            ('~/rgb/camera_info',  '/camera/color/camera_info'),
            ('~/stereo/image_raw', '/camera/aligned_depth_to_color/image_raw'),
        ],
    )

    # ── 7. CAS-GCS bin-picking pipeline node ──────────────────────────────
    pipeline = ExecuteProcess(
        cmd=[
            'python3', os.path.join(SCRIPTS_DIR, 'bin_picking_pipeline_node.py'),
            '--ros-args',
            '-p', f'checkpoint:={CNN_CHECKPOINT}',
            '-p', f'place_offset_y:={place_offset.perform(context)}',
            '-p', 'use_sim:=false',
        ],
        name='bin_picking_pipeline',
        output='screen',
    )

    # ── 8. Workspace collision scene (table + open-top bins) ──────────────
    # Runs as a persistent Node that re-publishes every 2 s so MoveIt always
    # has the collision objects even if it restarts.
    workspace_scene = ExecuteProcess(
        cmd=['python3', os.path.join(SCRIPTS_DIR, 'ur10_workspace_scene.py')],
        name='workspace_scene',
        output='screen',
    )

    # ── 9. Robotiq 3F gripper driver (Modbus TCP @ 192.168.1.105) ──────────
    gripper_driver = Node(
        package='robotiq_3f_gripper_ros2_driver',
        executable='robotiq_3f_gripper_node',
        name='robotiq_3f_gripper',
        output='screen',
        parameters=[{
            'gripper_ip'  : GRIPPER_IP,
            'gripper_port': 502,
            'speed'       : 150,
            'force'       : 100,
            'state_rate'  : 20.0,
        }],
        remappings=[
            # Publish gripper finger joints on a separate topic, NOT /joint_states.
            # The UR driver already publishes UR10 joints on /joint_states.
            # If finger joints (finger_1_joint_1 etc.) are merged into /joint_states,
            # move_group stores them in its robot state. When RViz requests the planning
            # scene it tries to set those variables on the plain UR10 model (which has no
            # gripper joints) → MoveIt throws "Variable not known" → RViz crashes.
            # The gripper is driven via Modbus/GripperCommand action, not MoveIt,
            # so move_group never needs to know about finger joint positions.
            ('joint_states', '/gripper/joint_states'),
        ],
    )


    # ── 9. TF static publishers ────────────────────────────────────────────
    # The UR driver's robot_state_publisher (ur_control.launch.py) covers all
    # UR10 joints.  We add TWO static publishers for the custom links that are
    # NOT in the UR driver's plain URDF:
    #
    # (a) wrist_3_link → wrist_camera_mount_link
    #     From URDF joint "wrist_camera_mount_joint":
    #       origin xyz="0.070 0.000 -0.100"  rpy="0 0 0"
    #     (bracket bolted 70 mm +X, 100 mm above flange -Z from wrist_3_link)
    wrist_mount_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='wrist_mount_tf',
        arguments=['0.070', '0.000', '-0.100', '0.0', '0.0', '0.0',
                   'wrist_3_link', 'wrist_camera_mount_link'],
    )

    # (b) wrist_camera_mount_link → camera_color_optical_frame
    #     x=0.03: camera centre is 10 cm from gripper centre in wrist X;
    #             URDF already offsets mount by 7 cm, so 3 cm more here.
    #     pitch=+1.5708: camera optical axis points straight down (along approach direction).
    #     NOTE: pitch=-1.5708 was wrong — it made the depth axis point UP (+Z in base_link),
    #     projecting objects to Z≈1.2m (above the robot) instead of Z≈0.07m (bin top).
    #     Flip to +1.5708 so camera_Z = mount +X = wrist_3_link +X = base_link -Z (down).
    wrist_cam_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='wrist_cam_tf',
        arguments=['0.03', '0.0', '0.0', '0.0', '1.5708', '0.0',
                   'wrist_camera_mount_link', 'camera_color_optical_frame'],
    )
    info = LogInfo(msg=[
        '\n╔══════════════════════════════════════════════════════════╗\n',
        '║   CAS-GCS BIN-PICKING  —  REAL HARDWARE  (ur_ws)        ║\n',
        '╠══════════════════════════════════════════════════════════╣\n',
        f'║  UR10 CB3    : {UR_ROBOT_IP}                           ║\n',
        f'║  Wrist OAK-D : {WRIST_CAM_MXID[:12]}... (USB 3.x)        ║\n',
        '║  CNN weights : /home/user/dataset/final_model.pth       ║\n',
        '╠══════════════════════════════════════════════════════════╣\n',
        '║  Trigger:  ros2 topic pub --once /pipeline/trigger \\    ║\n',
        '║            std_msgs/msg/Bool "data: true"               ║\n',
        '╚══════════════════════════════════════════════════════════╝\n',
    ])

    return [
        kill_stale,           # t=0: kill stale gripper/pipeline nodes from previous runs
        info,
        moveit,               # t=0: move_group loads our custom URDF from xacro directly
        wrist_mount_tf,       # static TF: wrist_3_link → wrist_camera_mount_link (from URDF)
        wrist_cam_tf,         # static TF: wrist_camera_mount_link → camera_color_optical_frame
        # NOTE: No rsp here — UR driver's RSP (ur_control.launch.py) is the sole RSP.
        #       Two RSPs competing causes "Link does not exist" in RViz and TF tree splits.
        TimerAction(period=3.0,  actions=[gripper_driver]), # t=3s:  gripper (after stale nodes killed)
        # t=5s: delay camera start — if camera_node starts at t=0 it enters a retry loop
        #        that locks the USB device (X_LINK_UNBOOTED) and never connects.
        #        A 5 s delay lets the USB bus settle after launch init.
        TimerAction(period=5.0,  actions=[wrist_cam]),
        TimerAction(period=15.0, actions=[
            workspace_scene,
            pipeline,
        ]),
        TimerAction(period=20.0, actions=[moveit_rviz]),  # t=20s: RViz last — after all topics ready
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ur_type',          default_value='ur10'),
        DeclareLaunchArgument('launch_rviz',       default_value='true'),
        DeclareLaunchArgument('place_offset_y',    default_value='0.60'),
        DeclareLaunchArgument('moveit_config_pkg', default_value='ur10_camera_moveit_config',
                              description='MoveIt config package: ur10_camera_moveit_config OR ur10_camera_gripper_moveit_config'),
        OpaqueFunction(function=launch_setup),
    ])
