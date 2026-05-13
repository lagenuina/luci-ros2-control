from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='luci_ros2_control',
            executable='pid_velocity',
            name='luci_pid_velocity_controller',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='luci_ros2_control',
            executable='pid_position',
            name='luci_position_pid',
            output='screen',
            emulate_tty=True,
        ),
    ])
