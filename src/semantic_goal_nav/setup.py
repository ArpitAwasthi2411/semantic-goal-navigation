from glob import glob

from setuptools import find_packages, setup

package_name = 'semantic_goal_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arpit Awasthi',
    maintainer_email='awasthiarpit24@gmail.com',
    description='Semantic goal navigation: YOLOv8n + semantic map + LLM commands + Nav2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector = semantic_goal_nav.yolo_detector:main',
            'semantic_map_manager = semantic_goal_nav.semantic_map_manager:main',
            'llm_command = semantic_goal_nav.llm_command:main',
            'nav2_goal_sender = semantic_goal_nav.nav2_goal_sender:main',
            'chat = semantic_goal_nav.chat:main',
        ],
    },
)
