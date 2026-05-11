#!/usr/bin/env python3
"""
luci_position_pid.py
---------------------
Outer-loop position PID controller for the LUCI wheelchair.

Subscribes to /luci/odom for actual position, computes distance and heading
error to a goal (x, y, yaw), and publishes a Twist to /cmd_vel.

Feed /cmd_vel into luci_pid_velocity_controller.py for the full cascade:

  desired (x,y,yaw) ──► [ Position PID ] ──► /cmd_vel ──► [ Velocity PID ] ──► LUCI
                                ▲
                           /luci/odom pose

Operates as a two-phase state machine:
  Phase 1 — Navigate to (goal_x, goal_y)
  Phase 2 — Rotate in-place to goal_yaw

Parameters:
    goal_x                (float, default  1.0)   meters
    goal_y                (float, default  0.0)   meters
    goal_yaw              (float, default  0.0)   radians
    goal_tolerance        (float, default  0.1)   meters
    orientation_tolerance (float, default  0.05)  radians (~3°)

    pos_kp           (float, default 1.0)
    pos_ki           (float, default 0.0)
    pos_kd           (float, default 0.0)

    heading_kp       (float, default 1.5)
    heading_ki       (float, default 0.0)
    heading_kd       (float, default 0.0)

    max_linear       (float, default 2.68)  m/s
    max_angular      (float, default 1.5)   rad/s
    windup_limit     (float, default 2.0)
    control_rate     (float, default 20.0)  Hz
    odom_topic       (string, default 'luci/odom')
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

def quat_to_yaw(q) -> float:
    """Extract yaw from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float, windup_limit: float):
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
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

class LuciPositionPID(Node):

    def __init__(self):
        super().__init__('luci_position_pid')

        self.declare_parameter('goal_x',                  0.0)
        self.declare_parameter('goal_y',                  -1.0)
        self.declare_parameter('goal_yaw',                0.0)
        self.declare_parameter('goal_tolerance',          0.05)
        self.declare_parameter('orientation_tolerance',   0.05)

        self.declare_parameter('pos_kp',         1.2)
        self.declare_parameter('pos_ki',         0.5)
        self.declare_parameter('pos_kd',         0.2)

        # self.declare_parameter('heading_kp',     1.23)
        # self.declare_parameter('heading_ki',     0.1) #0.1
        # self.declare_parameter('heading_kd',     0.08)

        self.declare_parameter('heading_kp',     1.25)
        self.declare_parameter('heading_ki',     0.02)
        self.declare_parameter('heading_kd',     0.01)

        self.declare_parameter('max_linear',     2.68)
        self.declare_parameter('max_angular',    1.5)
        self.declare_parameter('windup_limit',   2.0)
        self.declare_parameter('control_rate',   20.0)
        self.declare_parameter('odom_topic',     'luci/odom')

        p = lambda name: self.get_parameter(name).value

        self.goal_x               = p('goal_x')
        self.goal_y               = p('goal_y')
        self.goal_yaw             = p('goal_yaw')
        self.goal_tolerance       = p('goal_tolerance')
        self.orientation_tolerance = p('orientation_tolerance')
        self.max_linear           = p('max_linear')
        self.max_angular          = p('max_angular')

        # Two-phase flags: position first, then orientation
        self.goal_reached         = False   # Phase 1 complete
        self.orientation_reached  = False   # Phase 2 complete

        self.pid_position = PID(
            kp=p('pos_kp'), ki=p('pos_ki'), kd=p('pos_kd'),
            output_limit=self.max_linear, windup_limit=p('windup_limit')
        )
        self.pid_heading = PID(
            kp=p('heading_kp'), ki=p('heading_ki'), kd=p('heading_kd'),
            output_limit=self.max_angular, windup_limit=p('windup_limit')
        )

        self.actual_x      = 0.0
        self.actual_y      = 0.0
        self.actual_yaw    = 0.0
        self.last_time     = self.get_clock().now()
        self.last_odom_time = self.get_clock().now()
        self.odom_received = False

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.create_subscription(
            Odometry, p('odom_topic'), self.odom_cb, 5
        )

        self.create_timer(1.0 / p('control_rate'), self.control_loop)

        self.get_logger().info(
            f'Position PID started\n'
            f'  Goal   : x={self.goal_x} m  y={self.goal_y} m  '
            f'yaw={math.degrees(self.goal_yaw):.1f}°  '
            f'tol_pos={self.goal_tolerance} m  '
            f'tol_hdg={math.degrees(self.orientation_tolerance):.1f}°\n'
            f'  Pos PID: kp={p("pos_kp")}  ki={p("pos_ki")}  kd={p("pos_kd")}\n'
            f'  Hdg PID: kp={p("heading_kp")}  ki={p("heading_ki")}  kd={p("heading_kd")}'
        )

    def odom_cb(self, msg: Odometry):
        now = self.get_clock().now()
        dt  = (now - self.last_odom_time).nanoseconds * 1e-9
        self.last_odom_time = now

        # LUCI's forward velocity is on the y-axis of the twist message
        vx = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        # Integrate position via dead-reckoning
        self.actual_yaw += wz * dt
        self.actual_yaw  = angle_wrap(self.actual_yaw)
        self.actual_x   += vx * math.cos(self.actual_yaw) * dt
        self.actual_y   += vx * math.sin(self.actual_yaw) * dt

        # print(self.actual_yaw)

        self.odom_received = True

    def control_loop(self):
        if not self.odom_received:
            self.get_logger().warn(
                'Waiting for odometry...', throttle_duration_sec=2.0
            )
            return

        # Both phases done — nothing left to do
        if self.orientation_reached:
            return

        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # Phase 1: Navigate to (goal_x, goal_y)
        if not self.goal_reached:
            dx            = self.goal_x - self.actual_x
            dy            = self.goal_y - self.actual_y
            distance      = math.sqrt(dx**2 + dy**2)
            angle_to_goal = math.atan2(dy, dx)
            heading_error = angle_wrap(angle_to_goal - self.actual_yaw)

            if distance < self.goal_tolerance:
                self.get_logger().info(
                    f'Position reached!  '
                    f'x={self.actual_x:.2f}  y={self.actual_y:.2f}'
                )
                self.goal_reached = True
                # Reset heading PID so accumulated integral from navigation
                # does not cause overshoot during the orientation phase.
                self.pid_position.reset()
                self.pid_heading.reset()
                self.cmd_pub.publish(Twist())   # brief stop between phases
                return

            forward_scale = math.cos(heading_error)

            if forward_scale < 0.0:
                # Goal is behind — steer from the rear
                steer_error = angle_wrap(heading_error - math.pi)
            else:
                steer_error = heading_error

            desired_linear  = (self.pid_position.compute(distance, 0.0, dt)
                               * forward_scale)
            # desired_linear  = self.pid_position.compute(distance, 0.0, dt)
            desired_angular  = self.pid_heading.compute(steer_error, 0.0, dt)

            desired_linear  = max(-self.max_linear,
                                  min(self.max_linear,  desired_linear))
            desired_angular = max(-self.max_angular,
                                  min(self.max_angular, desired_angular))

            msg = Twist()
            msg.linear.x  = desired_linear
            msg.angular.z = desired_angular
            self.cmd_pub.publish(msg)

            self.get_logger().info(
                f'[Phase 1] dist={distance:.2f} m  '
                f'hdg_err={math.degrees(heading_error):.1f}°  '
                f'lin={desired_linear:.2f} m/s  '
                f'ang={desired_angular:.2f} rad/s',
                throttle_duration_sec=0.5
            )
            return

        # Phase 2: Rotate in-place to goal_yaw
        yaw_error = angle_wrap(self.goal_yaw - self.actual_yaw)

        if abs(yaw_error) < self.orientation_tolerance:
            self.get_logger().info(
                f'Orientation reached!  '
                f'yaw={math.degrees(self.actual_yaw):.1f}°'
            )
            self.orientation_reached = True
            self.cmd_pub.publish(Twist())   # full stop
            return

        msg = Twist()
        msg.angular.z = self.pid_heading.compute(yaw_error, 0.0, dt)
        self.cmd_pub.publish(msg)

        self.get_logger().info(
            f'[Phase 2] yaw_err={math.degrees(yaw_error):.1f}°  '
            f'ang={msg.angular.z:.2f} rad/s',
            throttle_duration_sec=0.5
        )

    def set_goal(self, x: float, y: float, yaw: float = 0.0):
        """Set a new goal position + orientation and reset all PIDs."""
        self.goal_x              = x
        self.goal_y              = y
        self.goal_yaw            = yaw
        self.goal_reached        = False
        self.orientation_reached = False
        self.pid_position.reset()
        self.pid_heading.reset()
        self.get_logger().info(
            f'New goal → x={x} m  y={y} m  yaw={math.degrees(yaw):.1f}°'
        )

    def destroy_node(self):
        self.get_logger().info('Shutting down — publishing stop.')
        self.cmd_pub.publish(Twist())
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LuciPositionPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()