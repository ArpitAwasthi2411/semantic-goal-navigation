"""Gazebo with the AWS small house and a TurtleBot3 Waffle Pi (360 deg LiDAR + RGB camera).

    ros2 launch semantic_goal_nav sim.launch.py            # with the Gazebo window
    ros2 launch semantic_goal_nav sim.launch.py gui:=false # headless

Start order matters on a laptop: gzserver loads the house (can take a minute), then the robot
is spawned (waiting up to 5 min for Gazebo), and only then the Gazebo window (gzclient) opens.
Opening gzclient while the world is still loading is what makes gzclient 11.10 abort.
"""
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, RegisterEventHandler,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

os.environ.setdefault('TURTLEBOT3_MODEL', 'waffle_pi')


def gazebo_setup_env(script='/usr/share/gazebo/setup.sh'):
    """The GAZEBO_* / OGRE_* variables that Gazebo's setup.sh exports ({} if it is not there)."""
    if not os.path.isfile(script):
        return {}
    try:
        out = subprocess.run(['bash', '-c', f'. {script} >/dev/null 2>&1; env -0'],
                             capture_output=True, timeout=10).stdout.decode()
    except (OSError, subprocess.SubprocessError):
        return {}
    env = {}
    for item in out.split('\0'):
        k, _, v = item.partition('=')
        if k.startswith(('GAZEBO_', 'OGRE_')) and k != 'GAZEBO_MODEL_DATABASE_URI':
            env[k] = v
    return env


def generate_launch_description():
    gazebo_ros = get_package_share_directory('gazebo_ros')
    tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    house = get_package_share_directory('aws_robomaker_small_house_world')
    model = os.environ['TURTLEBOT3_MODEL']
    robot_sdf = os.path.join(tb3_gazebo, 'models', f'turtlebot3_{model}', 'model.sdf')

    gui = LaunchConfiguration('gui')
    world = LaunchConfiguration('world')
    x, y, yaw = LaunchConfiguration('x_pose'), LaunchConfiguration('y_pose'), LaunchConfiguration('yaw')

    def paths(*ps):
        return os.pathsep.join(p for p in ps if p)
    # Gazebo's own environment (/usr/share/gazebo/setup.sh): resource, plugin and OGRE paths.
    # Without OGRE_RESOURCE_PATH / GAZEBO_RESOURCE_PATH gzclient can abort at start-up.
    gz_env = gazebo_setup_env()
    env = [SetEnvironmentVariable(k, v) for k, v in gz_env.items() if k != 'GAZEBO_MODEL_PATH']
    env += [
        SetEnvironmentVariable('TURTLEBOT3_MODEL', model),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', paths(
            os.path.join(house, 'models'), os.path.join(tb3_gazebo, 'models'),
            os.environ.get('GAZEBO_MODEL_PATH', ''), gz_env.get('GAZEBO_MODEL_PATH', ''))),
        # no online model database: every model is local, and the lookup only slows start-up
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
    ]

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-entity', model, '-file', robot_sdf, '-x', x, '-y', y, '-z', '0.01', '-Y', yaw,
                            '-timeout', '300'])
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(gui))

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='show the Gazebo window'),
        DeclareLaunchArgument('world', default_value=os.path.join(house, 'worlds', 'small_house.world')),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        *env,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world, 'verbose': 'false'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[
            LogInfo(msg='[semantic_goal_nav] robot spawned; opening the Gazebo window'),
            TimerAction(period=3.0, actions=[gzclient])])),
    ])
