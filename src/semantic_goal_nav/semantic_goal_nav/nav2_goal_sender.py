#!/usr/bin/env python3
"""⑤ Nav2 Goal Sender node.

In:  /semantic_goal   geometry_msgs/PoseStamped (from the LLM command node, or RViz/CLI)
     /semantic_goal_nav/cancel  std_msgs/Empty
     /map             (to move a goal that lands in/next to an obstacle onto free floor)
Out: Nav2 'navigate_to_pose' action goals
     /semantic_goal_nav/nav_status  std_msgs/String JSON:
        {"goal_id": 3, "state": "accepted|moving|recovering|succeeded|failed|canceled",
         "goal": [x, y], "sent": [x, y], "distance": 1.2, "message": "..."}

On an aborted goal it runs a recovery (back up a little, then rotate in place through the
Nav2 behaviour servers) and sends the goal again, up to `max_retries` times.
It can also set AMCL's initial pose at start-up (publish_initial_pose:=true), so the robot
is localised without clicking "2D Pose Estimate" in RViz.
"""
import json
import math
import threading

import rclpy
import rclpy.executors
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import BackUp, NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from .geometry import OccupancyGrid, yaw_to_quaternion

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class Nav2GoalSender(Node):
    def __init__(self):
        super().__init__('nav2_goal_sender')
        dyn = ParameterDescriptor(dynamic_typing=True)   # launch may pass 0 (int) or 0.0
        p = lambda name, value: self.declare_parameter(name, value, dyn)  # noqa: E731
        p('goal_topic', '/semantic_goal')
        p('robot_radius', 0.22)
        p('goal_clearance_margin', 0.08)
        p('max_retries', 1)
        p('backup_distance', 0.15)
        p('publish_initial_pose', False)
        p('initial_x', 0.0)
        p('initial_y', 0.0)
        p('initial_yaw', 0.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.min_clear = float(g('robot_radius')) + float(g('goal_clearance_margin'))
        self.max_retries = int(g('max_retries'))
        self.backup_dist = float(g('backup_distance'))

        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.backup_client = ActionClient(self, BackUp, 'backup')
        self.status_pub = self.create_publisher(String, '/semantic_goal_nav/nav_status', 10)
        self.create_subscription(PoseStamped, g('goal_topic'), self._on_goal, 10)
        self.create_subscription(Empty, '/semantic_goal_nav/cancel', lambda _: self.cancel('canceled by user'), 10)
        self.create_subscription(OccupancyGridMsg, '/map', self._on_map, LATCHED)

        self.grid = None
        self.lock = threading.Lock()
        self.goal_id = 0
        self.active = None      # dict: id, goal(PoseStamped), handle, retries, requested xy
        self.last_fb_log = 0.0

        if g('publish_initial_pose'):
            self.init_pose = (float(g('initial_x')), float(g('initial_y')), float(g('initial_yaw')))
            self.amcl_seen = False
            self.init_count = 0
            self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl, LATCHED)
            self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
            self.init_timer = self.create_timer(2.0, self._publish_initial_pose)
        self.get_logger().info('ready: send geometry_msgs/PoseStamped to ' + g('goal_topic'))

    # ------------------------------------------------------------------ initial pose
    def _on_amcl(self, msg):
        # localised = AMCL reports a pose close to the one we gave it (after we sent it)
        if self.init_count == 0:
            return
        x, y, _ = self.init_pose
        p = msg.pose.pose.position
        if math.hypot(p.x - x, p.y - y) < 0.5:
            self.amcl_seen = True

    def _publish_initial_pose(self):
        if self.amcl_seen:
            self.get_logger().info('AMCL is localised at the initial pose')
            self.init_timer.cancel()
            return
        if self.init_count >= 30:
            self.get_logger().warn('AMCL did not take the initial pose; set it with "2D Pose Estimate" in RViz')
            self.init_timer.cancel()
            return
        if self.count_subscribers('/initialpose') == 0:
            return      # AMCL not up yet
        x, y, yaw = self.init_pose
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        # stamp 0 = "use the latest transform": a current stamp can be a few ms ahead of the
        # odom TF in simulation, and AMCL then drops the pose ("extrapolation into the future")
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        q = yaw_to_quaternion(yaw)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = q[2], q[3]
        m.pose.covariance[0] = m.pose.covariance[7] = 0.05
        m.pose.covariance[35] = 0.05
        self.init_pub.publish(m)
        self.init_count += 1
        if self.init_count == 1 or self.init_count % 5 == 0:
            self.get_logger().info(f'initial pose sent to AMCL: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)')

    # ------------------------------------------------------------------ goals
    def _on_map(self, msg):
        self.grid = OccupancyGrid.from_msg(msg)

    def _status(self, state, message='', distance=None):
        a = self.active
        d = {'goal_id': a['id'] if a else self.goal_id, 'state': state, 'message': message}
        if a:
            d['goal'] = [round(a['requested'][0], 3), round(a['requested'][1], 3)]
            d['sent'] = [round(a['goal'].pose.position.x, 3), round(a['goal'].pose.position.y, 3)]
        if distance is not None:
            d['distance'] = round(float(distance), 2)
        self.status_pub.publish(String(data=json.dumps(d)))
        log = self.get_logger().info if state in ('accepted', 'succeeded', 'moving') else self.get_logger().warn
        log(f'[goal {d["goal_id"]}] {state} {message}'.rstrip())

    def _on_goal(self, msg):
        if not msg.header.frame_id:
            msg.header.frame_id = 'map'
        req = (msg.pose.position.x, msg.pose.position.y)
        x, y = req
        note = ''
        if self.grid is not None:
            free = self.grid.nearest_clear(x, y, self.min_clear, max_search=1.5)
            if free is None:
                free = self.grid.nearest_clear(x, y, self.min_clear * 0.7, max_search=2.0)
            if free is not None and math.hypot(free[0] - x, free[1] - y) > 0.02:
                note = f'(moved {math.hypot(free[0] - x, free[1] - y):.2f} m onto free floor)'
                msg.pose.position.x, msg.pose.position.y = free
        with self.lock:
            if self.active and self.active.get('handle'):
                self.active['handle'].cancel_goal_async()
            self.goal_id += 1
            self.active = {'id': self.goal_id, 'goal': msg, 'handle': None, 'retries': 0, 'requested': req}
        self.get_logger().info(f'[goal {self.goal_id}] to ({msg.pose.position.x:.2f}, '
                               f'{msg.pose.position.y:.2f}) {note}'.rstrip())
        # in its own thread: _send may wait for localisation, and callbacks must keep running
        threading.Thread(target=self._send, args=(self.goal_id,), daemon=True).start()

    def _send(self, gid):
        if not self.nav.wait_for_server(timeout_sec=5.0):
            self._status('failed', 'Nav2 is not running (no navigate_to_pose action server)')
            return
        # A goal sent before AMCL is localised has no map->odom transform behind it, and Nav2
        # then repeats "extrapolation into the past" forever. Hold it until localised.
        if getattr(self, 'amcl_seen', True) is False:
            self._status('waiting', 'waiting for localisation before driving')
            t0 = self.get_clock().now().nanoseconds
            while not self.amcl_seen and rclpy.ok():
                if (self.get_clock().now().nanoseconds - t0) * 1e-9 > 120.0:
                    self._status('failed', 'robot is not localised; set "2D Pose Estimate" in RViz')
                    return
                threading.Event().wait(0.5)
        with self.lock:
            if not self.active or self.active['id'] != gid:
                return
            goal = NavigateToPose.Goal()
            goal.pose = self.active['goal']
            # stamp 0 = "transform with the latest data". A real stamp goes stale on a slow
            # laptop (simulation slower than real time) and Nav2 cannot transform it any more.
            goal.pose.header.stamp = Time().to_msg()
        fut = self.nav.send_goal_async(goal, feedback_callback=lambda fb: self._on_feedback(gid, fb))
        fut.add_done_callback(lambda f: self._on_response(gid, f))

    def _on_response(self, gid, fut):
        try:
            handle = fut.result()
        except Exception as e:  # noqa: BLE001
            self._status('failed', f'goal request failed: {e}')
            return
        with self.lock:
            if not self.active or self.active['id'] != gid:
                handle.cancel_goal_async()
                return
            if not handle.accepted:
                self._status('failed', 'Nav2 rejected the goal')
                self.active = None
                return
            self.active['handle'] = handle
        self._status('accepted')
        handle.get_result_async().add_done_callback(lambda f: self._on_result(gid, f))

    def _on_feedback(self, gid, fb):
        if not self.active or self.active['id'] != gid:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_fb_log >= 2.0:
            self.last_fb_log = now
            self._status('moving', distance=fb.feedback.distance_remaining)

    def _on_result(self, gid, fut):
        try:
            status = fut.result().status
        except Exception as e:  # noqa: BLE001
            status, err = None, str(e)
        else:
            err = ''
        with self.lock:
            if not self.active or self.active['id'] != gid:
                return          # an old goal that was replaced
            a = self.active
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._status('succeeded', 'destination reached')
            self.active = None
        elif status == GoalStatus.STATUS_CANCELED:
            self._status('canceled')
            self.active = None
        elif a['retries'] < self.max_retries:
            a['retries'] += 1
            a['handle'] = None
            self._status('recovering', f'Nav2 aborted the goal {err}; backing up, rotating and retrying '
                                       f'({a["retries"]}/{self.max_retries})'.replace('  ', ' '))
            threading.Thread(target=self._recover_then_retry, args=(gid,), daemon=True).start()
        else:
            self._status('failed', 'could not reach the destination' + (f' ({err})' if err else ''))
            self.active = None

    def _recover_then_retry(self, gid):
        self._run_behavior(self.backup_client, self._backup_goal(), 8.0)
        self._run_behavior(self.spin_client, self._spin_goal(), 12.0)
        if self.active and self.active['id'] == gid:
            self._send(gid)

    def _backup_goal(self):
        g = BackUp.Goal()
        g.target.x = -abs(self.backup_dist)
        g.speed = 0.08
        g.time_allowance = Duration(seconds=6.0).to_msg()
        return g

    def _spin_goal(self):
        g = Spin.Goal()
        g.target_yaw = math.pi / 2
        g.time_allowance = Duration(seconds=10.0).to_msg()
        return g

    def _run_behavior(self, client, goal, timeout):
        if not client.wait_for_server(timeout_sec=2.0):
            return
        ev = threading.Event()

        def on_resp(f):
            try:
                h = f.result()
            except Exception:  # noqa: BLE001
                ev.set()
                return
            if not h.accepted:
                ev.set()
                return
            h.get_result_async().add_done_callback(lambda _r: ev.set())
        client.send_goal_async(goal).add_done_callback(on_resp)
        ev.wait(timeout)

    def cancel(self, why):
        with self.lock:
            a = self.active
            if a is None:
                return
            if a.get('handle'):
                a['handle'].cancel_goal_async()
            self._status('canceled', why)
            self.active = None


def main(args=None):
    rclpy.init(args=args)
    node = Nav2GoalSender()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
