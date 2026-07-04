from glob import glob
import os

from setuptools import setup

package_name = 'luci_ros2_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lagenuina',
    maintainer_email='genua.l@northeastern.edu',
    description='PID-based velocity and position controllers for the LUCI wheelchair.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pid_velocity = luci_ros2_control.pid_velocity:main',
            'pid_position = luci_ros2_control.pid_position:main',
            'local_planner = luci_ros2_control.local_planner:main',
            'global_planner = luci_ros2_control.global_planner:main',
            'twist_to_luci_joystick = luci_ros2_control.twist_to_luci_joystick:main',
        ],
    },
)
