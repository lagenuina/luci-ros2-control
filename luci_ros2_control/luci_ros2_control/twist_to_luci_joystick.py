#!/usr/bin/env python3
"""Twist -> LuciJoystick converter.

Subscribes to a Twist command (default /cmd_vel) and translates each
(linear.x, angular.z) into a LUCI remote-joystick deflection using a
per-speed-profile inverse cubic and publishes on luci/remote_joystick.

"""

from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from luci_messages.msg import LuciJoystick
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32
from std_srvs.srv import Empty

CMD_VEL_TOPIC       = '/cmd_vel'
SPEED_SETTING_TOPIC = '/luci/chair_profile'
JOYSTICK_TOPIC      = '/luci/remote_joystick'

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)

# Per-profile inverse cubics: deflection = a*v^3 + b*v^2 + c*v
# Fit from measured (deflection, speed) data for each LUCI speed profile.
# Key = profile number (1-5), value = (a, b, c).
PROFILE_CUBICS: dict[int, tuple[float, float, float]] = {
    1: ( -26.8986,   75.8661, 151.7333),
    2: ( -15.7565,   29.9407,  78.6467),
    3: (   2.9095,   -3.4397,  61.9742),
    4: (   3.2353,   -6.1976,  48.5822),
    5: (  -0.6564,    1.6430,  36.4053),
}

# Maximum speed (m/s) at full deflection per profile — used only for warnings.
PROFILE_MAX_SPEED = {1: 0.536, 2: 1.073, 3: 1.565, 4: 2.012, 5: 2.772}

# Per-profile angular linear fit: w (rad/s) = slope * lr + intercept  (symmetric for negative)
# Fit from measured (deflection, angular_speed) data.
# Inverse: lr = (w - intercept) / slope
# Key = profile number (1-5), value = (slope, intercept).
# PROFILE_ANGULAR_LINEAR: dict[int, tuple[float, float]] = {
#     1: (0.002168, 0.014901),
#     2: (0.003009, 0.026822),
#     3: (0.003334, 0.026822),
#     4: (0.004470, 0.000000),
#     5: (0.005122, 0.026822),
# }

class Zone:
    FRONT       = 0
    FRONT_LEFT  = 1
    FRONT_RIGHT = 2
    LEFT        = 3
    RIGHT       = 4
    BACK_LEFT   = 5
    BACK_RIGHT  = 6
    BACK        = 7
    ORIGIN      = 8


class InputSource:
    REMOTE = 1


def compute_zone(fb: int, lr: int) -> int:
    if fb == 0 and lr == 0:
        return Zone.ORIGIN
    if fb > 0:
        if lr < 0: return Zone.FRONT_LEFT
        if lr > 0: return Zone.FRONT_RIGHT
        return Zone.FRONT
    if fb < 0:
        if lr < 0: return Zone.BACK_LEFT
        if lr > 0: return Zone.BACK_RIGHT
        return Zone.BACK
    return Zone.LEFT if lr < 0 else Zone.RIGHT


def _clamp_deflection(x: float) -> int:
    """Clamp a joystick deflection to the [-100, 100] integer range."""
    return int(max(-100.0, min(100.0, x)))


class TwistToJoystickNode(Node):

    def __init__(self) -> None:
        super().__init__('twist_to_joystick')

        self._speed_setting: Optional[int] = None

        self.create_subscription(
            Twist, 
            CMD_VEL_TOPIC, 
            self._on_cmd_vel, 
            50)
        self.create_subscription(
            Int32,
            SPEED_SETTING_TOPIC,
            self._on_speed_setting,
            LATCHED_QOS)

        self.pub_joy = self.create_publisher(
            LuciJoystick, 
            JOYSTICK_TOPIC, 
            50)

        # Take over remote control so the chair acts on luci/remote_joystick.
        self._call_service('/luci/set_auto_remote_input')

        self.get_logger().info('twist_to_joystick node started.')

    def _on_speed_setting(self, msg: Int32) -> None:
        if msg.data != self._speed_setting:
            self.get_logger().info(f'LUCI speed profile -> {msg.data}.')
        self._speed_setting = int(msg.data)

    def _vel_to_deflection(self, v: float) -> Optional[int]:
        """Convert a desired speed (m/s) to a joystick deflection using the
        active profile's fitted inverse cubic.  Returns None if the profile
        is unknown so the caller can drop the command safely."""
        if self._speed_setting is None:
            self.get_logger().warn(
                'Speed profile not yet received — dropping cmd_vel.',
                throttle_duration_sec=2.0,
            )
            return None
        coeffs = PROFILE_CUBICS.get(self._speed_setting)
        if coeffs is None:
            self.get_logger().warn(
                f'No cubic calibration for profile {self._speed_setting} — dropping cmd_vel.',
                throttle_duration_sec=2.0,
            )
            return None
        a, b, c = coeffs
        fb = a * v**3 + b * v**2 + c * v
        v_max = PROFILE_MAX_SPEED.get(self._speed_setting, float('inf'))
        if v > v_max:
            self.get_logger().warn(
                f'Requested v={v:.2f} m/s exceeds profile {self._speed_setting} '
                f'max ({v_max:.1f} m/s); deflection will saturate.',
                throttle_duration_sec=5.0,
            )
        return _clamp_deflection(fb)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._publish_joystick(msg.linear.x, msg.angular.z)

    def _publish_stop(self) -> None:
        js = LuciJoystick()
        js.forward_back  = 0
        js.left_right    = 0
        js.joystick_zone = Zone.ORIGIN
        js.input_source  = InputSource.REMOTE
        self.pub_joy.publish(js)

    def _publish_joystick(self, v: float, w: float) -> None:
        if v == 0.0:
            fb = 0
        else:
            result = self._vel_to_deflection(v)
            if result is None:
                return
            fb = result
        
        if w == 0.0:
            lr = 0
        else:
            lr = _clamp_deflection((w - 0.0676) / -0.012212)

        js = LuciJoystick()
        js.forward_back  = fb
        js.left_right    = lr
        js.joystick_zone = compute_zone(fb, lr)
        js.input_source  = InputSource.REMOTE
        self.pub_joy.publish(js)

    def _call_service(self, service_name: str) -> None:
        client = self.create_client(Empty, service_name)
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f'Service {service_name} not available — '
                'is luci_grpc_interface_node running?'
            )
            return
        future = client.call_async(Empty.Request())
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'Service {service_name} called successfully.'
            )
        )

    def destroy_node(self) -> None:

        self.get_logger().info('Shutting down — stopping chair.')
        self._publish_stop()
        self._call_service('/luci/remove_auto_remote_input')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TwistToJoystickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
