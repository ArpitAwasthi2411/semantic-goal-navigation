#!/bin/bash
# One-time setup on Ubuntu 22.04 with ROS 2 Humble already installed (/opt/ros/humble).
#   cd ~/semantic_goal_nav_ws && bash setup.sh
set -e
source /opt/ros/humble/setup.bash

echo "== ROS packages (Gazebo, TurtleBot3, Nav2, Cartographer) =="
sudo apt update
sudo apt install -y \
  ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3 ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-msgs \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-cartographer ros-humble-cartographer-ros \
  ros-humble-rviz2 ros-humble-tf2-ros python3-pip python3-scipy python3-yaml python3-pil python3-colcon-common-extensions

echo "== YOLOv8 (Ultralytics). numpy<2 keeps the ROS Humble Python packages working =="
pip3 install --user "numpy<2" ultralytics

echo "== Build =="
cd "$(dirname "$0")"
colcon build --symlink-install
grep -q 'semantic_goal_nav_ws/install/setup.bash' ~/.bashrc || {
  echo "source $(pwd)/install/setup.bash" >> ~/.bashrc
  echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
}

echo "== Optional: local LLM =="
echo "  curl -fsSL https://ollama.com/install.sh | sh && ollama pull gemma2:2b"
echo "  (without it the robot uses keyword mode, which already understands the commands)"
echo
echo "Done. Open a new terminal and run:  ros2 launch semantic_goal_nav semantic_goal_nav.launch.py"
