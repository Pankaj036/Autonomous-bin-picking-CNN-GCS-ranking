## UR10 Autonomous Bin Picking System

ROS2 Humble based autonomous bin-picking system using UR10 robot, OAK-D cameras, MoveIt2, visual servoing, AI grasp planning, and Robotiq 3F gripper.

---

## Features

- Autonomous bin picking
- Real UR10 robot support
- Gazebo simulation support
- MoveIt2 integration
- Visual servoing
- AI grasp planning
- OAK-D stereo depth cameras
- Robotiq 3F gripper support
- Voice command control using LLM
- Collision-aware manipulation
- Continuous pick-and-place pipeline
- Real-time RGB + depth processing
- CNN-based object detection
- TF2 coordinate transformations

---

## Hardware Used

- UR10 Robot
- Robotiq 3F Gripper
- Robotiq FT300 Sensor
- OAK-D Pro Wide Camera
- OAK-D Pro AF Camera
- Ubuntu 22.04
- ROS2 Humble

---

## Software Stack

- ROS2 Humble
- MoveIt2
- Gazebo
- OpenCV
- Python
- NumPy
- RTDE
- TF2
- DepthAI
- CNN Grasp Planning

---

## Repository Structure

```bash
Pick-Place-Simulation/
│
├── auto_bin_picking.py
├── bin_picking_pipeline_node.py
├── sim_visual_servoing_node.py
├── ur10_llm_command_node.py
├── sim_visual_servoing.launch.py
├── real_ur10_oakd_gripper_moveit.launch.py
├── images/
├── videos/
├── README.md
```

---

## System Architecture

## Real Hardware Pipeline

1. Environment camera detects object
2. CNN inference generates grasp candidates
3. Wrist camera captures RGB + depth
4. Visual servoing aligns robot
5. MoveIt2 plans collision-free trajectory
6. Robotiq gripper grasps object
7. Object placed in destination tray

---

## Camera Setup

## Environment Camera
- OAK-D Pro AF
- Fixed workspace camera

## Wrist Camera
- OAK-D Pro Wide
- Mounted near gripper

---

## ROS2 Topics

## Subscribed Topics

```bash
/env_camera/rgb/image_raw
/wrist_camera/rgb/image_raw
/pipeline/trigger
/joint_states
```

## Published Topics

```bash
/grasp_pose
/place_pose
/pipeline/status
/pre_grasp_scan_pose
```

---

## Installation

## Clone Repository

```bash
git clone https://github.com/Pankaj036/Pick-Place-Simulation.git
```

---

## Build Workspace

```bash
cd ~/ur_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Run Simulation

## Launch Gazebo + MoveIt

```bash
ros2 launch ur_yt_sim sim_visual_servoing.launch.py
```

---

## Run Simulation Visual Servoing

```bash
python3 sim_visual_servoing_node.py
```

---

## Run Real Hardware

## Launch Real UR10 + OAK-D + Gripper

```bash
ros2 launch ur_yt_sim real_ur10_oakd_gripper_moveit.launch.py
```

---

## Run Bin Picking Pipeline

```bash
python3 bin_picking_pipeline_node.py
```

---

## Run Automatic Pipeline Trigger

```bash
python3 auto_bin_picking.py
```

---

## Voice Command Support

Run:

```bash
python3 ur10_llm_command_node.py
```

Supported commands:
- pick blue object
- track red object
- pause tracking
- resume
- clear selection

---

## Robot Configuration

```python
UR10 Robot IP : 192.168.1.102
Gripper IP    : 192.168.1.105
PC IP         : 192.168.1.10
```

---

## Visual Servoing Features

- HSV color tracking
- Real-time object alignment
- Contour filtering
- Wrist orientation correction
- Dynamic servo control
- Automatic grasp execution

---

## AI Grasp Planning

The system uses:
- CNN-based segmentation
- Depth estimation
- Grasp candidate scoring
- Collision-aware planning
- 3D grasp pose estimation

---

## Gazebo Simulation

Simulation includes:
- UR10 robot
- Robotiq gripper
- OAK-D camera
- MoveIt2 planning
- RViz visualization
- Object spawning

---

## Future Improvements

- YOLOv8 integration
- Reinforcement learning
- Multi-object sorting
- VLA training
- Dynamic obstacle avoidance
- Isaac Sim integration

### For real robot launch – and external control play button on
bash
cd ur_ws
ros2 launch ur_robot_driver ur_control.launch.py \
ur_type:=ur10 \
robot_ip:=192.168.1.102 \
kinematics_config:="${HOME}/my_ur10_calibration.yaml" \
launch_rviz:=false

### bash2
cd ur_ws/src
colcon build --symlink-install
source install/setup.bash
ros2 launch ur_yt_sim spawn_ur10_camera_gripper_moveit.launch.py

### for real robot 
cd ur_ws/src
colcon build --symlink-install
source install/setup.bash
ros2 launch ur_yt_sim real_ur10_3f_gripper_moveit_launch.py 

### 3 f gripper Simulation (Gazebo + RViz2 + MoveIt):
  ros2 launch ur_yt_sim spawn_ur10_camera_3f_gripper_moveit.launch.py

  
