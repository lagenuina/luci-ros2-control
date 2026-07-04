from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='luci_ros2_control',
            executable='local_planner',
            name='local_planner',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='luci_ros2_control',
            executable='twist_to_luci_joystick',
            name='twist_to_luci_joystick',
            output='screen',
            emulate_tty=True,
        ),
    ])
