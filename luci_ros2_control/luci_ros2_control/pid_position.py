#!/usr/bin/env python3
import math

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry

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
        d = 0.0 if self._prev_measure is None else \
            -self.kd * (measurement - self._prev_measure) / dt
        self._prev_measure = measurement
        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral     = 0.0
        self._prev_measure = None

class LuciPositionPID(Node):

    def __init__(self):
        super().__init__('luci_position_pid')

        self.declare_parameter('goal_tolerance',          0.05)
        self.declare_parameter('orientation_tolerance',   0.05)
        self.declare_parameter('heading_deadband',        0.26)
        self.declare_parameter('pose_topic',              'rachel')
        self.declare_parameter('pos_kp',                  1.2)
        self.declare_parameter('pos_ki',                  0.5)
        self.declare_parameter('pos_kd',                  0.2)
        self.declare_parameter('heading_kp',              1.25)
        self.declare_parameter('heading_ki',              0.02)
        self.declare_parameter('heading_kd',              0.01)
        self.declare_parameter('max_linear',              2.68)
        self.declare_parameter('max_angular',             1.5)
        self.declare_parameter('windup_limit',            2.0)
        self.declare_parameter('control_rate',            20.0)

        p = lambda name: self.get_parameter(name).value

        self.goal_tolerance        = p('goal_tolerance')
        self.orientation_tolerance = p('orientation_tolerance')
        self.heading_deadband      = p('heading_deadband')
        self.max_linear            = p('max_linear')
        self.max_angular           = p('max_angular')

        self.pid_position = PID(
            kp=p('pos_kp'), ki=p('pos_ki'), kd=p('pos_kd'),
            output_limit=self.max_linear, windup_limit=p('windup_limit')
        )
        self.pid_heading = PID(
            kp=p('heading_kp'), ki=p('heading_ki'), kd=p('heading_kd'),
            output_limit=self.max_angular, windup_limit=p('windup_limit')
        )

        self.actual_x       = 0.0
        self.actual_y       = 0.0
        self.actual_yaw     = 0.0
        self.last_time      = self.get_clock().now()
        self.pose_received  = False

        self.goal_x              = 0.0
        self.goal_y              = 0.0
        self.goal_yaw            = 0.0
        self.goal_reached        = False
        self.orientation_reached = False
        self._active_goal_handle = None

        self.world_frame = 'motive_world'
        self.robot_frame = p('pose_topic')  # default: rachel

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub  = self.create_publisher(Twist,       'cmd_vel',      10)
        self.goal_pub = self.create_publisher(PoseStamped, 'current_goal', 10)

        cb_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_callback,
            callback_group=cb_group,
        )

        self.create_timer(
            1.0 / p('control_rate'),
            self.control_loop,
            callback_group=cb_group
        )

        self.create_timer(
            0.01,
            self.update_pose,
            callback_group=cb_group
        )

        self.get_logger().info(
            f'LuciPositionPID ready\n'
            f'  Pose source : /{p("pose_topic")} (map frame, from RTAB-Map)\n'
            f'  Goal topic  : /current_goal  '
            f'(add a Pose display in RViz to see the arrow)\n'
            f'  Action      : /navigate_to_pose  '
            f'(send goals in map frame)'
        )


    def update_pose(self):
        """
        Lookup tf transform from 'motive_world' to 'rachel'
        and use it as the robot's current pose.
        """

        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.robot_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f'Waiting for TF {self.world_frame} -> {self.robot_frame}: {e}',
                throttle_duration_sec=2.0
            )
            return

        self.actual_x = tf.transform.translation.x
        self.actual_y = tf.transform.translation.y
        self.actual_yaw = quat_to_yaw(tf.transform.rotation)

        self.pose_received = True
 
    def _goal_callback(self, goal_request):
        self.get_logger().info('Received new navigation goal — accepting.')
        if self._active_goal_handle is not None:
            self.get_logger().info('Preempting previous goal.')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested.')
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        """
        Goal is already in map frame — set it directly, no TF needed.
        Publish it to /current_goal for RViz visualization.
        """
        pose = goal_handle.request.pose.pose

        goal_stamped = PoseStamped()
        goal_stamped.header.frame_id = 'map'
        goal_stamped.header.stamp    = self.get_clock().now().to_msg()
        goal_stamped.pose            = pose
        self.goal_pub.publish(goal_stamped)

        self._set_goal(
            x=pose.position.x,
            y=pose.position.y,
            yaw=quat_to_yaw(pose.orientation),
        )
        self._active_goal_handle = goal_handle

        feedback_msg = NavigateToPose.Feedback()
        rate = self.create_rate(20)

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal cancelled.')
                self._stop_robot()
                goal_handle.canceled()
                self._active_goal_handle = None
                return NavigateToPose.Result()


            if self._active_goal_handle is not goal_handle:
                self.get_logger().info('Goal preempted.')
                goal_handle.abort()
                return NavigateToPose.Result()


            if self.orientation_reached:
                self.get_logger().info('Goal succeeded.')
                goal_handle.succeed()
                self._active_goal_handle = None
                return NavigateToPose.Result()


            dx = self.goal_x - self.actual_x
            dy = self.goal_y - self.actual_y
            feedback_msg.distance_remaining = math.sqrt(dx**2 + dy**2)
            goal_handle.publish_feedback(feedback_msg)

            rate.sleep()

        goal_handle.abort()
        self._active_goal_handle = None
        return NavigateToPose.Result()

    def control_loop(self):
        if not self.pose_received:
            self.get_logger().warn(
                'Waiting for RTAB-Map pose on '
                f'/{self.get_parameter("pose_topic").value} ...',
                throttle_duration_sec=2.0
            )
            return

        if self._active_goal_handle is None or self.orientation_reached:
            return

        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

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
                self.pid_position.reset()
                self.pid_heading.reset()
                self._stop_robot()
                return

            # Rotate in place when not facing the goal, unless we're already
            # close enough that angle_to_goal becomes noisy.
            far_from_goal   = distance > 2.0 * self.goal_tolerance
            facing_goal     = abs(heading_error) <= self.heading_deadband

            if far_from_goal and not facing_goal:
                self.pid_position.reset()
                desired_linear  = 0.0
                desired_angular = self.pid_heading.compute(
                    heading_error, 0.0, dt
                )
            else:
                desired_linear  = self.pid_position.compute(distance, 0.0, dt)
                desired_angular = self.pid_heading.compute(
                    heading_error, 0.0, dt
                )

            # Forward-only: never command negative linear velocity.
            desired_linear  = max(0.0, min(self.max_linear, desired_linear))
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

        yaw_error = angle_wrap(self.goal_yaw - self.actual_yaw)

        if abs(yaw_error) < self.orientation_tolerance:
            self.get_logger().info(
                f'Orientation reached!  '
                f'yaw={math.degrees(self.actual_yaw):.1f}°'
            )
            self.orientation_reached = True
            self._stop_robot()
            return

        msg = Twist()
        msg.angular.z = self.pid_heading.compute(yaw_error, 0.0, dt)
        self.cmd_pub.publish(msg)

        self.get_logger().info(
            f'[Phase 2] yaw_err={math.degrees(yaw_error):.1f}°  '
            f'ang={msg.angular.z:.2f} rad/s',
            throttle_duration_sec=0.5
        )

    def _set_goal(self, x: float, y: float, yaw: float):
        self.goal_x              = x
        self.goal_y              = y
        self.goal_yaw            = yaw
        self.goal_reached        = False
        self.orientation_reached = False
        self.pid_position.reset()
        self.pid_heading.reset()
        self.last_time = self.get_clock().now()
        self.get_logger().info(
            f'New goal (map frame) → '
            f'x={x:.2f} m  y={y:.2f} m  yaw={math.degrees(yaw):.1f}°'
        )

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self.get_logger().info('Shutting down — publishing stop.')
        self._stop_robot()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LuciPositionPID()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()