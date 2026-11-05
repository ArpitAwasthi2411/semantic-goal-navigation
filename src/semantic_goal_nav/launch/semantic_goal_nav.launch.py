"""Everything in one launch: Gazebo house + TurtleBot3, localisation (AMCL on a saved map, or
Cartographer SLAM), Nav2, and the semantic navigation nodes.

    ros2 launch semantic_goal_nav semantic_goal_nav.launch.py                    # clean map of the house (default)
    ros2 launch semantic_goal_nav semantic_goal_nav.launch.py world_map:=old     # our earlier apartment_map (see README)
    ros2 launch semantic_goal_nav semantic_goal_nav.launch.py slam:=true         # Cartographer maps live
    ros2 launch semantic_goal_nav semantic_goal_nav.launch.py gui:=false         # no Gazebo window (RViz only)

Then give commands in a second terminal:  ros2 run semantic_goal_nav chat
"""
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo, \
    OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

os.environ.setdefault('TURTLEBOT3_MODEL', 'waffle_pi')


def world_to_map_pose(x, y, yaw, t):
    """Spawn pose in the Gazebo world -> the same pose in the map frame (for AMCL)."""
    a = math.radians(float(t.get('yaw_deg', 0.0)))
    dx, dy = x - float(t.get('x', 0.0)), y - float(t.get('y', 0.0))
    mx = math.cos(a) * dx + math.sin(a) * dy
    my = -math.sin(a) * dx + math.cos(a) * dy
    return mx, my, math.atan2(math.sin(yaw - a), math.cos(yaw - a))


def setup(context):
    pkg = get_package_share_directory('semantic_goal_nav')
    nav2 = get_package_share_directory('nav2_bringup')
    cfg = lambda f: os.path.join(pkg, 'config', f)  # noqa: E731
    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    truthy = lambda n: arg(n).lower() in ('true', '1', 'yes')  # noqa: E731

    with open(cfg('map_profiles.yaml')) as f:
        profiles = yaml.safe_load(f)
    name = arg('world_map')
    if name not in profiles:
        raise RuntimeError(f"world_map:={name} unknown; use one of {list(profiles)}")
    prof = profiles[name]
    slam = truthy('slam')

    spawn = dict(prof.get('spawn', {}))
    for k, a in (('x', 'x_pose'), ('y', 'y_pose'), ('yaw', 'yaw')):
        if arg(a) != '':
            spawn[k] = float(arg(a))
    sx, sy, syaw = float(spawn.get('x', 0)), float(spawn.get('y', 0)), float(spawn.get('yaw', 0))
    if slam:      # Cartographer starts its map where the robot starts: map frame = spawn pose
        ix, iy, iyaw = 0.0, 0.0, 0.0
        rooms_file, objects_file = '', ''
    else:
        ix, iy, iyaw = world_to_map_pose(sx, sy, syaw, prof.get('world_to_map', {}))
        rooms_file, objects_file = cfg(prof['rooms']), cfg(prof['known_objects'])
    map_yaml = arg('map') or os.path.join(pkg, 'maps', prof['map'])
    save_file = arg('semantic_map_file') or f'~/.ros/semantic_map_{"slam" if slam else name}.json'
    sim_time = {'use_sim_time': True}

    actions = [LogInfo(msg=f'[semantic_goal_nav] map profile "{name}"{" + Cartographer SLAM" if slam else ""}: '
                           f'spawn world ({sx:.2f}, {sy:.2f}, {math.degrees(syaw):.0f} deg) -> '
                           f'initial pose map ({ix:.2f}, {iy:.2f}, {math.degrees(iyaw):.0f} deg)')]

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'gui': arg('gui'), 'x_pose': str(sx), 'y_pose': str(sy), 'yaw': str(syaw)}.items()))

    if not slam:
        # ① AMCL on the saved map (Nav2 bringup = map_server + amcl + navigation)
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'bringup_launch.py')),
            launch_arguments={'slam': 'False', 'map': map_yaml, 'use_sim_time': 'true',
                              'params_file': cfg('nav2_params.yaml'), 'autostart': 'true',
                              'use_composition': 'True'}.items()))
    else:
        # ① Cartographer SLAM + Nav2 navigation only
        actions.append(GroupAction([
            Node(package='cartographer_ros', executable='cartographer_node', name='cartographer_node',
                 output='screen', parameters=[sim_time],
                 arguments=['-configuration_directory', os.path.join(pkg, 'config'),
                            '-configuration_basename', 'cartographer.lua']),
            Node(package='cartographer_ros', executable='cartographer_occupancy_grid_node',
                 name='cartographer_occupancy_grid_node', output='screen', parameters=[sim_time],
                 arguments=['-resolution', '0.05', '-publish_period_sec', '1.0']),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'navigation_launch.py')),
                launch_arguments={'use_sim_time': 'true', 'params_file': cfg('nav2_params.yaml'),
                                  'autostart': 'true', 'use_composition': 'True'}.items()),
        ]))

    nodes = [
        # ② YOLOv8n detector
        Node(package='semantic_goal_nav', executable='yolo_detector', name='yolo_detector', output='screen',
             parameters=[cfg('semantic_goal_nav.yaml'), sim_time, {'model': arg('yolo_model')}]),
        # ③ semantic map manager
        Node(package='semantic_goal_nav', executable='semantic_map_manager', name='semantic_map_manager',
             output='screen',
             parameters=[cfg('semantic_goal_nav.yaml'), sim_time, {
                 'rooms_file': rooms_file, 'known_objects_file': objects_file,
                 'use_known_objects': truthy('use_known_objects'), 'save_file': save_file,
                 'load_saved': truthy('load_saved_map')}]),
        # ⑤ Nav2 goal sender (also gives AMCL its initial pose on a saved map)
        Node(package='semantic_goal_nav', executable='nav2_goal_sender', name='nav2_goal_sender', output='screen',
             parameters=[cfg('semantic_goal_nav.yaml'), sim_time, {
                 'publish_initial_pose': not slam, 'initial_x': ix, 'initial_y': iy, 'initial_yaw': iyaw}]),
        # ④ LLM command node (commands from `ros2 run semantic_goal_nav chat` via /user_command)
        Node(package='semantic_goal_nav', executable='llm_command', name='llm_command', output='screen',
             parameters=[cfg('semantic_goal_nav.yaml'), sim_time, {
                 'mode': arg('llm_mode'), 'ollama_model': arg('ollama_model'),
                 'ollama_url': arg('ollama_url'), 'interactive': False}]),
    ]
    if truthy('rviz'):
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
                          arguments=['-d', os.path.join(pkg, 'rviz', 'semantic_goal_nav.rviz')], parameters=[sim_time]))
    actions.append(TimerAction(period=5.0, actions=nodes))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_map', default_value='house',
                              description='house = clean map of the house; old = our earlier apartment_map '
                                          'with the team\'s rooms/objects (robot gets lost on it, see README)'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='true: Cartographer maps live; false: AMCL on the saved map'),
        DeclareLaunchArgument('map', default_value='', description='override the map yaml of the profile'),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='', description='spawn x in the Gazebo world'),
        DeclareLaunchArgument('y_pose', default_value=''),
        DeclareLaunchArgument('yaw', default_value=''),
        DeclareLaunchArgument('llm_mode', default_value='keyword', description='auto | llm | keyword'),
        DeclareLaunchArgument('ollama_model', default_value='gemma2:2b'),
        DeclareLaunchArgument('ollama_url', default_value='http://localhost:11434'),
        DeclareLaunchArgument('yolo_model', default_value='yolov8n.pt',
                              description='yolov8n.pt (COCO) or a path to your trained weights'),
        DeclareLaunchArgument('use_known_objects', default_value='true',
                              description='preload the objects marked for this map'),
        DeclareLaunchArgument('semantic_map_file', default_value='',
                              description='default ~/.ros/semantic_map_<world_map>.json'),
        DeclareLaunchArgument('load_saved_map', default_value='true',
                              description='reload objects detected in earlier runs'),
        OpaqueFunction(function=setup),
    ])
