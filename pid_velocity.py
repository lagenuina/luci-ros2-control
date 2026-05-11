#!/usr/bin/env python3
"""
luci_pid_velocity_controller.py
--------------------------------
Combined PID velocity controller + LuciJoystick publisher for the LUCI wheelchair.

Reads actual velocity from LUCI's odometry, runs dual PID (linear + angular),
and publishes directly to /luci/remote_joystick — no intermediate /cmd_vel needed.

                 ┌──────────────────────────────┐
desired_vel ───► │   PID  →  joystick scaling   │ ──► /luci/remote_joystick ──► LUCI
                 │        (this node)            │ ◄── /luci/odom
                 └──────────────────────────────┘

Tunable parameters (ROS2 params or launch file):
    desired_linear   (float, default 0.5)   m/s
    desired_angular  (float, default 0.0)   rad/s

    linear_kp  (float, default 1.0)
    linear_ki  (float, default 0.1)
    linear_kd  (float, default 0.05)

    angular_kp (float, default 1.5)
    angular_ki (float, default 0.1)
    angular_kd (float, default 0.05)

    max_linear   (float, default 2.68)  m/s   -- 6 mph hard cap
    max_angular  (float, default 1.5)   rad/s
    windup_limit (float, default 1.0)
    control_rate (float, default 20.0)  Hz
    negate_lr    (bool,  default True)  -- flip angular sign for LUCI convention
    odom_topic   (string, default 'luci/odom')
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from luci_messages.msg import LuciJoystick
from geometry_msgs.msg import Twist

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


def scale(value: float, max_value: float) -> int:
    """Scale a physical value to [-100, 100] and clamp."""
    if max_value == 0.0:
        return 0
    return int(max(-100.0, min(100.0, (value / max_value) * 100.0)))

class PID:
    """
    PID with anti-windup integral clamping and derivative on measurement
    (avoids derivative kick on setpoint changes).
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float, windup_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.windup_limit = windup_limit

        self._integral     = 0.0
        self._prev_measure = None

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        error = setpoint - measurement

        p = self.kp * error

        self._integral += error * dt
        self._integral  = max(-self.windup_limit,
                              min(self.windup_limit, self._integral))
        i = self.ki * self._integral

        if self._prev_measure is None:
            d = 0.0
        else:
            d = -self.kd * (measurement - self._prev_measure) / dt
        self._prev_measure = measurement

        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral     = 0.0
        self._prev_measure = None


class LuciPIDVelocityController(Node):

    def __init__(self):
        super().__init__('luci_pid_velocity_controller')

        # self.declare_parameter('desired_linear',  2)
        # self.declare_parameter('desired_angular', 0.0)

        self.declare_parameter('linear_kp',  1.2)
        self.declare_parameter('linear_ki',  0.5)
        self.declare_parameter('linear_kd',  0.2)

        self.declare_parameter('angular_kp', 1.5)
        self.declare_parameter('angular_ki', 0.1)
        self.declare_parameter('angular_kd', 0.05)

        self.declare_parameter('max_linear',   2.68)
        self.declare_parameter('max_angular',  1.5)
        self.declare_parameter('windup_limit', 2.0)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('negate_lr',    True)
        self.declare_parameter('odom_topic',   'luci/odom')

        p = lambda name: self.get_parameter(name).value

        self.desired_linear  = 0
        self.desired_angular = 0
        self.max_linear      = p('max_linear')
        self.max_angular     = p('max_angular')
        self.negate_lr       = p('negate_lr')
        rate_hz              = p('control_rate')
        odom_topic           = p('odom_topic')

        self.pid_linear = PID(
            kp=p('linear_kp'),  ki=p('linear_ki'),  kd=p('linear_kd'),
            output_limit=self.max_linear,  windup_limit=p('windup_limit')
        )
        self.pid_angular = PID(
            kp=p('angular_kp'), ki=p('angular_ki'), kd=p('angular_kd'),
            output_limit=self.max_angular, windup_limit=p('windup_limit')
        )

        self.actual_linear  = 0.0
        self.actual_angular = 0.0
        self.last_time      = self.get_clock().now()
        self.odom_received  = False

        self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_cb, 10)

        self.pub = self.create_publisher(
            LuciJoystick, 'luci/remote_joystick', 10
        )

        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)

        self.create_timer(1.0 / rate_hz, self.control_loop)

        self._call_service('/luci/set_auto_remote_input')

        self.get_logger().info(
            f'LUCI PID velocity controller started\n'
            f'  Target : linear={self.desired_linear} m/s  '
            f'angular={self.desired_angular} rad/s\n'
            f'  Linear  PID: kp={p("linear_kp")}  '
            f'ki={p("linear_ki")}  kd={p("linear_kd")}\n'
            f'  Angular PID: kp={p("angular_kp")}  '
            f'ki={p("angular_ki")}  kd={p("angular_kd")}\n'
            f'  Odom   : {odom_topic}'
        )

    def cmd_vel_cb(self, msg: Twist):
        self.desired_linear  = msg.linear.x
        self.desired_angular = msg.angular.z
        self.pid_linear.reset()
        self.pid_angular.reset()

    def odom_cb(self, msg: Odometry):
        self.actual_linear  = msg.twist.twist.linear.y
        self.actual_angular = msg.twist.twist.angular.z
        self.odom_received  = True

        print(msg.twist.twist.linear.y)

    def control_loop(self):
        if not self.odom_received:
            self.get_logger().warn(
                'Waiting for odometry...', throttle_duration_sec=2.0
            )
            return

        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        linear_cmd  = self.pid_linear.compute(
            self.desired_linear, self.actual_linear, dt
        )
        angular_cmd = self.pid_angular.compute(
            self.desired_angular, self.actual_angular, dt
        )

        fb = scale(linear_cmd, self.max_linear)
        lr = scale(
            -angular_cmd if self.negate_lr else angular_cmd,
            self.max_angular
        )

        js = LuciJoystick()
        js.forward_back  = fb
        js.left_right    = lr
        js.joystick_zone = compute_zone(fb, lr)
        js.input_source  = InputSource.REMOTE
        self.pub.publish(js)

        self.get_logger().info(
            f'lin: set={self.desired_linear:.2f}  '
            f'act={self.actual_linear:.2f}  '
            f'cmd={linear_cmd:.2f}  js={fb}  |  '
            f'ang: set={self.desired_angular:.2f}  '
            f'act={self.actual_angular:.2f}  '
            f'cmd={angular_cmd:.2f}  js={lr}',
            throttle_duration_sec=0.5
        )

    def set_desired(self, linear: float, angular: float):
        """Update target velocity. Resets integral to avoid windup carry-over."""
        self.desired_linear  = linear
        self.desired_angular = angular
        self.pid_linear.reset()
        self.pid_angular.reset()
        self.get_logger().info(
            f'New target → linear={linear} m/s  angular={angular} rad/s'
        )

    def _call_service(self, service_name: str):
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

    def destroy_node(self):
        """Send a stop command and restore physical joystick on shutdown."""
        self.get_logger().info('Shutting down — stopping chair.')
        js = LuciJoystick()
        js.forward_back  = 0
        js.left_right    = 0
        js.joystick_zone = Zone.ORIGIN
        js.input_source  = InputSource.REMOTE
        self.pub.publish(js)
        self._call_service('/luci/remove_auto_remote_input')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LuciPIDVelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()