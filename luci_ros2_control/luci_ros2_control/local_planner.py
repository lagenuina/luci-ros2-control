#!/usr/bin/env python3
"""
Standalone MPPI test: subscribe to /goal_pose and drive there directly.
No global path, no risk map — pure MPPI with terminal-distance and
heading-alignment costs only.
"""

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from numba import njit, prange
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException,
                     LookupException, TransformListener)

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


@njit(cache=True, fastmath=True, inline='always')
def _bilinear_risk(x, y, risk, ox, oy, res, h, w, mu_blocked):
    cx = (x - ox) / res - 0.5
    cy = (y - oy) / res - 0.5
    ix = int(math.floor(cx))
    iy = int(math.floor(cy))
    if ix < 0 or iy < 0 or ix + 1 >= w or iy + 1 >= h:
        return mu_blocked
    fx = cx - ix
    fy = cy - iy
    r00 = risk[iy, ix]
    r10 = risk[iy, ix + 1]
    r01 = risk[iy + 1, ix]
    r11 = risk[iy + 1, ix + 1]
    return ((1.0 - fy) * ((1.0 - fx) * r00 + fx * r10) +
            fy * ((1.0 - fx) * r01 + fx * r11))


@njit(cache=True, fastmath=True, parallel=True)
def _mppi_costs(
    x0, y0, yaw0,
    V_nom, W_nom,
    V_noise, W_noise,
    risk, ox, oy, res, h_grid, w_grid, mu_blocked,
    dt, v_max, w_min, w_max,
    target_x, target_y, terminal_w, heading_w,
):
    K, H = V_noise.shape
    costs = np.zeros(K)
    V_used = np.empty((K, H))
    W_used = np.empty((K, H))
    for k in prange(K):
        x = x0
        y = y0
        th = yaw0
        c = 0.0
        for t in range(H):
            v = V_nom[t] + V_noise[k, t]
            w = W_nom[t] + W_noise[k, t]
            if v > v_max:
                v = v_max
            elif v < 0.0:
                v = 0.0
            if w > w_max:
                w = w_max
            elif w < -w_max:
                w = -w_max
            V_used[k, t] = v
            W_used[k, t] = w
            w_phys = 0.0 if -w_min < w < w_min else w
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w_phys * dt
            c += _bilinear_risk(x, y, risk, ox, oy, res,
                                h_grid, w_grid, mu_blocked)
        dx = x - target_x
        dy = y - target_y
        c += terminal_w * math.sqrt(dx * dx + dy * dy)
        bearing = math.atan2(-dy, -dx)
        herr = math.atan2(math.sin(th - bearing), math.cos(th - bearing))
        c += heading_w * herr * herr
        costs[k] = c
    return costs, V_used, W_used


@njit(cache=True, fastmath=True)
def _rollout_traj(x0, y0, yaw0, V_nom, W_nom, dt, v_min, v_max, w_max):
    H = V_nom.shape[0]
    traj = np.empty((H + 1, 2))
    traj[0, 0] = x0
    traj[0, 1] = y0
    x = x0
    y = y0
    th = yaw0
    for t in range(H):
        v = V_nom[t]
        w = W_nom[t]
        if v > v_max:
            v = v_max
        elif v < v_min:
            v = v_min
        if w > w_max:
            w = w_max
        elif w < -w_max:
            w = -w_max
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        traj[t + 1, 0] = x
        traj[t + 1, 1] = y
    return traj


class State:
    IDLE     = 'idle'
    REACHING = 'reaching'
    ALIGNING = 'aligning'


class MppiGoalTestNode(Node):

    def __init__(self) -> None:
        super().__init__('mppi_goal_test')

        self.declare_parameter('num_samples',        1000)
        self.declare_parameter('horizon',            40)
        self.declare_parameter('mppi_dt',            0.1)
        self.declare_parameter('sigma_v',            0.12)
        self.declare_parameter('sigma_omega',        0.1)
        self.declare_parameter('lambda_',            0.3)
        self.declare_parameter('v_min',              0.15)
        self.declare_parameter('v_max',              0.4)
        self.declare_parameter('w_max',              0.4)
        self.declare_parameter('w_min',              0.10)
        self.declare_parameter('terminal_w',         12.0)
        self.declare_parameter('heading_w',          0.5)
        self.declare_parameter('publish_rate',       15.0)
        self.declare_parameter('map_frame',          'map')
        self.declare_parameter('robot_frame',        'base_link')
        self.declare_parameter('goal_tolerance',     0.3)
        self.declare_parameter('heading_tolerance',  0.2)
        self.declare_parameter('heading_k',          1.0)
        self.declare_parameter('heading_w_max',      0.4)
        self.declare_parameter('obstacle_topic',     '/octomap_grid')
        self.declare_parameter('obstacle_cost',      1e6)
        self.declare_parameter('obstacle_lookahead',  0.4)

        self._goal:          Optional[PoseStamped]   = None
        self._obstacle_grid: Optional[OccupancyGrid] = None
        self._state: str                             = State.IDLE

        H = int(self.get_parameter('horizon').value)
        self._V_nom = np.zeros(H, dtype=np.float64)
        self._W_nom = np.zeros(H, dtype=np.float64)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseStamped, '/goal_pose',
                                 self._on_goal, 10)
        self.create_subscription(
            OccupancyGrid, self.get_parameter('obstacle_topic').value,
            self._on_obstacle, LATCHED_QOS)

        self.pub_cmd    = self.create_publisher(Twist,       '/cmd_vel',      10)
        self.pub_target = self.create_publisher(PoseStamped, '/target_pose',  10)

        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info('mppi_goal_test node started — send a /goal_pose to begin.')

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._V_nom.fill(0.0)
        self._W_nom.fill(0.0)
        self._state = State.REACHING
        self.get_logger().info(
            f'New goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

    def _on_obstacle(self, msg: OccupancyGrid) -> None:
        self._obstacle_grid = msg

    def _build_cost_grid(self, obs: OccupancyGrid) -> np.ndarray:
        sentinel = float(self.get_parameter('obstacle_cost').value)
        raw = np.frombuffer(bytes(obs.data), dtype=np.int8).reshape(
            obs.info.height, obs.info.width
        )
        grid = np.zeros((obs.info.height, obs.info.width), dtype=np.float64)
        grid[raw >= 50] = sentinel
        return grid

    def _robot_pose(self) -> Optional[Tuple[float, float, float]]:
        map_frame   = self.get_parameter('map_frame').value
        robot_frame = self.get_parameter('robot_frame').value
        try:
            t = self.tf_buffer.lookup_transform(
                map_frame, robot_frame, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {map_frame}<-{robot_frame} unavailable: {e}',
                throttle_duration_sec=5.0,
            )
            return None
        qx = t.transform.rotation.x
        qy = t.transform.rotation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def _mppi_step(
        self,
        pose: Tuple[float, float, float],
        target: Tuple[float, float],
    ) -> Tuple[float, float]:
        K      = int(self.get_parameter('num_samples').value)
        H      = int(self.get_parameter('horizon').value)
        dt     = float(self.get_parameter('mppi_dt').value)
        sig_v  = float(self.get_parameter('sigma_v').value)
        sig_w  = float(self.get_parameter('sigma_omega').value)
        lam    = float(self.get_parameter('lambda_').value)
        v_min  = float(self.get_parameter('v_min').value)
        v_max  = float(self.get_parameter('v_max').value)
        w_min  = float(self.get_parameter('w_min').value)
        w_max  = float(self.get_parameter('w_max').value)
        term_w = float(self.get_parameter('terminal_w').value)
        head_w = float(self.get_parameter('heading_w').value)

        V_noise = np.random.normal(0.0, sig_v, size=(K, H))
        W_noise = np.random.normal(0.0, sig_w, size=(K, H))

        if self._obstacle_grid is not None:
            obs         = self._obstacle_grid
            risk_grid   = self._build_cost_grid(obs)
            grid_ox     = float(obs.info.origin.position.x)
            grid_oy     = float(obs.info.origin.position.y)
            grid_res    = float(obs.info.resolution)
            grid_h      = int(obs.info.height)
            grid_w      = int(obs.info.width)
            mu_blk      = 0.0  # no terrain risk; outside grid = free
        else:
            # No obstacle data yet — dummy 1×1 grid that contributes nothing.
            risk_grid = np.zeros((1, 1), dtype=np.float64)
            grid_ox, grid_oy, grid_res = 0.0, 0.0, 1e6
            grid_h, grid_w, mu_blk = 1, 1, 0.0

        costs, V_used, W_used = _mppi_costs(
            pose[0], pose[1], pose[2],
            self._V_nom, self._W_nom,
            V_noise, W_noise,
            risk_grid, grid_ox, grid_oy, grid_res, grid_h, grid_w, mu_blk,
            dt, v_max, w_min, w_max,
            target[0], target[1], term_w, head_w,
        )

        rho = float(np.min(costs))
        w = np.exp(-(costs - rho) / lam)
        w_sum = float(np.sum(w))
        if w_sum < 1e-12:
            w = np.ones(K) / K
        else:
            w /= w_sum

        # Weighted average of the *actually simulated* (clamped) controls —
        # averaging the raw noise instead would drift the nominal away from
        # what was costed, since samples saturate to the same rollout.
        self._V_nom = (w[:, None] * V_used).sum(axis=0)
        self._W_nom = (w[:, None] * W_used).sum(axis=0)

        raw_v = float(self._V_nom[0])
        raw_w = float(self._W_nom[0])

        blocked = self._path_blocked(pose, risk_grid, grid_ox, grid_oy,
                                     grid_res, grid_h, grid_w, mu_blk)

        if blocked:
            v_cmd = 0.0
        elif raw_v < v_min:
            v_cmd = v_min
        else:
            v_cmd = min(raw_v, v_max)

        if abs(raw_w) < w_min:
            w_cmd = 0.0
        else:
            w_cmd = float(np.clip(raw_w, -w_max, w_max))

        self._V_nom = np.concatenate([self._V_nom[1:], self._V_nom[-1:]])
        self._W_nom = np.concatenate([self._W_nom[1:], self._W_nom[-1:]])

        return v_cmd, w_cmd

    def _path_blocked(
        self,
        pose: Tuple[float, float, float],
        risk: np.ndarray, ox: float, oy: float, res: float,
        h: int, w: int, mu_blocked: float,
    ) -> bool:
        if self._obstacle_grid is None:
            return False
        lookahead = float(self.get_parameter('obstacle_lookahead').value)
        lx = pose[0] + lookahead * math.cos(pose[2])
        ly = pose[1] + lookahead * math.sin(pose[2])
        risk_ahead = _bilinear_risk(lx, ly, risk, ox, oy, res, h, w, mu_blocked)
        sentinel = float(self.get_parameter('obstacle_cost').value)
        return risk_ahead >= sentinel / 2.0

    def _publish_stop(self) -> None:
        self.pub_cmd.publish(Twist())

    def _publish_cmd(self, v: float, w: float) -> None:
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

    def _tick(self) -> None:
        if self._state == State.IDLE:
            self._publish_stop()
            return

        pose = self._robot_pose()
        if pose is None:
            self._publish_stop()
            return

        # ── ALIGNING ──────────────────────────────────────────────────────────
        if self._state == State.ALIGNING:
            q = self._goal.pose.orientation
            desired_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            herr = math.atan2(math.sin(pose[2] - desired_yaw),
                              math.cos(pose[2] - desired_yaw))
            head_tol = float(self.get_parameter('heading_tolerance').value)
            if abs(herr) <= head_tol:
                self._state = State.IDLE
                self._publish_stop()
                self.get_logger().info('[tick] goal reached — going idle.')
                return
            head_k     = float(self.get_parameter('heading_k').value)
            head_w_max = float(self.get_parameter('heading_w_max').value)
            w_cmd = float(np.clip(-head_k * herr, -head_w_max, head_w_max))
            self._publish_cmd(0.0, w_cmd)
            self.get_logger().info(
                f'[tick] aligning: herr={math.degrees(herr):.1f} deg  w={w_cmd:.2f}',
                throttle_duration_sec=1.0,
            )
            return

        # ── REACHING ──────────────────────────────────────────────────────────
        target_xy = (self._goal.pose.position.x, self._goal.pose.position.y)
        dist = math.hypot(pose[0] - target_xy[0], pose[1] - target_xy[1])
        tol  = float(self.get_parameter('goal_tolerance').value)
        if dist <= tol:
            self._state = State.ALIGNING
            return

        v_cmd, w_cmd = self._mppi_step(pose, target_xy)
        self._publish_cmd(v_cmd, w_cmd)

        now = self.get_clock().now().to_msg()
        tps = PoseStamped()
        tps.header.stamp    = now
        tps.header.frame_id = self.get_parameter('map_frame').value
        tps.pose.position.x = float(target_xy[0])
        tps.pose.position.y = float(target_xy[1])
        tps.pose.orientation.w = 1.0
        self.pub_target.publish(tps)

        self.get_logger().info(
            f'[tick] cmd=(v={v_cmd:.2f}, w={w_cmd:.2f}) dist={dist:.2f} m',
            throttle_duration_sec=1.0,
        )

    def destroy_node(self) -> None:
        self.get_logger().info('Shutting down — stopping chair.')
        self._publish_stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MppiGoalTestNode()
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
