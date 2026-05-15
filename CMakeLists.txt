cmake_minimum_required(VERSION 3.8)
project(ur_yt_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(linkattacher_msgs REQUIRED)  
find_package(ros2_linkattacher REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(test_ik src/test_ik.cpp)
ament_target_dependencies(test_ik rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs)

add_executable(test_ik_collision_obj src/test_ik_collision_obj.cpp)
ament_target_dependencies(test_ik_collision_obj rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs)

add_executable(pick_place src/pick_place.cpp)
ament_target_dependencies(pick_place rclcpp moveit_ros_planning_interface sensor_msgs geometry_msgs linkattacher_msgs ros2_linkattacher)


install(DIRECTORY launch urdf config meshes worlds rviz models scripts
  DESTINATION share/${PROJECT_NAME})

# Python entry-points (ros2 run ur_yt_sim <node>)
install(PROGRAMS scripts/cnn_grasp_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cnn_grasp_node)

install(PROGRAMS scripts/bin_pick_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bin_pick_node)

install(PROGRAMS scripts/camera_quad_view.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_quad_view)

install(PROGRAMS scripts/bin_picking_pipeline_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME bin_picking_pipeline_node)

install(PROGRAMS scripts/cnn_vis_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cnn_vis_node)

install(TARGETS 
  test_ik test_ik_collision_obj
  test_ik test_ik_collision_obj pick_place
  DESTINATION lib/${PROJECT_NAME}/
)


ament_package()