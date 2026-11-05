#!/usr/bin/env python3
"""④ LLM Command node.

Turns a sentence into a goal:
  text -> steps (Ollama LLM, or keyword matching) -> semantic map lookup -> approach point
       -> /semantic_goal (PoseStamped) -> Nav2 Goal Sender -> waits for the result -> next step

Commands come from
  /user_command  std_msgs/String  (the `chat` console publishes here)
  this terminal, when started with interactive:=true (default when run by hand in a terminal)
Replies go to /semantic_goal_nav/feedback (std_msgs/String) and to the log.

mode: auto    LLM when Ollama answers, keyword matching otherwise (and whenever the LLM
              reply is unusable)  <- default
      llm     LLM only
      keyword keyword matching only (no Ollama needed)
"""
import json
import math
import queue
import sys
import threading
import time

import rclpy
import rclpy.executors
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener

from .command_parser import (build_prompt, describe_step, extract_json, keyword_parse,
                             validate_llm_steps)
from .geometry import OccupancyGrid, choose_approach, quaternion_to_yaw, yaw_to_quaternion
from .llm_client import OllamaClient
from .semantic_map import SemanticMap

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)

# Rough footprint radius of furniture: the stop point must not be on top of it (e.g. under the
# table when going to a chair on its far side).
OBJECT_RADIUS = {'dining table': 0.8, 'coffee table': 0.6, 'bed': 1.1, 'couch': 0.9, 'desk': 0.6,
                 'wardrobe': 0.6, 'cabinet': 0.5, 'refrigerator': 0.5, 'shelf': 0.4}


class LlmCommandNode(Node):
    def __init__(self, interactive=None):
        super().__init__('llm_command')
        dyn = ParameterDescriptor(dynamic_typing=True)   # accept 1 as well as 1.0 from YAML/launch
        p = lambda name, value: self.declare_parameter(name, value, dyn)  # noqa: E731
        p('mode', 'auto')
        p('ollama_url', 'http://localhost:11434')
        p('ollama_model', 'gemma2:2b')
        p('llm_timeout', 60.0)
        p('standoff', 0.6)                 # stop this far from the object
        p('goal_timeout', 180.0)
        p('max_candidates', 3)             # try the next-nearest object if one is unreachable
        p('map_frame', 'map')
        p('base_frame', 'base_footprint')
        p('interactive', bool(interactive) if interactive is not None else sys.stdin.isatty())
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = g('mode')
        self.standoff = float(g('standoff'))
        self.goal_timeout = float(g('goal_timeout'))
        self.max_candidates = int(g('max_candidates'))
        self.map_frame, self.base_frame = g('map_frame'), g('base_frame')
        self.llm = OllamaClient(g('ollama_url'), g('ollama_model'), float(g('llm_timeout')))
        self.llm_ok = None

        self.smap = SemanticMap()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_pub = self.create_publisher(PoseStamped, '/semantic_goal', 10)
        self.cancel_pub = self.create_publisher(Empty, '/semantic_goal_nav/cancel', 10)
        self.fb_pub = self.create_publisher(String, '/semantic_goal_nav/feedback', 10)
        self.create_subscription(String, '/semantic_map', self._on_map, LATCHED)
        self.grid = None
        self.create_subscription(OccupancyGridMsg, '/map', self._on_grid, LATCHED)
        self.create_subscription(String, '/semantic_goal_nav/nav_status', self._on_status, 10)
        self.create_subscription(String, '/user_command', lambda m: self.submit(m.data), 10)

        self.commands = queue.Queue()
        self.nav_event = threading.Event()
        self.nav_result = None
        self.waiting_for = None       # (x, y) of the goal we sent
        self.abort = threading.Event()
        self.last_progress = 0.0
        threading.Thread(target=self._worker, daemon=True).start()
        if g('interactive'):
            threading.Thread(target=self._stdin_loop, daemon=True).start()
        threading.Thread(target=self._check_llm, daemon=True).start()

    # ------------------------------------------------------------------ io
    def say(self, text):
        self.get_logger().info(text)
        self.fb_pub.publish(String(data=text))

    def submit(self, text):
        text = (text or '').strip()
        if not text:
            return
        # "stop" acts immediately, even while a goal is running
        quick = keyword_parse(text, self.smap.room_names())
        if len(quick) == 1 and quick[0]['action'] == 'stop':
            self._stop()
            return
        self.commands.put(text)

    def _stdin_loop(self):
        time.sleep(1.0)
        print('\nType a command (e.g. "go to the bed", "take me to the kitchen", '
              '"what do you see", "stop"). Ctrl+C to quit.', flush=True)
        while rclpy.ok():
            try:
                text = input('command> ')
            except (EOFError, KeyboardInterrupt):
                return
            self.submit(text)

    def _check_llm(self):
        if self.mode == 'keyword':
            self.say('keyword mode: no LLM needed')
            return
        self.llm_ok = self.llm.available()
        if self.llm_ok:
            self.say(f'LLM mode: Ollama model {self.llm.model} at {self.llm.url}')
        else:
            self.say(f'Ollama ({self.llm.model}) not reachable at {self.llm.url}: '
                     + ('using keyword mode' if self.mode == 'auto' else 'commands will fail (mode=llm)'))

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg):
        try:
            self.smap = SemanticMap.from_dict(json.loads(msg.data))
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(f'bad /semantic_map message: {e}')

    def _on_grid(self, msg):
        self.grid = OccupancyGrid.from_msg(msg)

    def _on_status(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if self.waiting_for is None or 'goal' not in d:
            return
        if math.hypot(d['goal'][0] - self.waiting_for[0], d['goal'][1] - self.waiting_for[1]) > 0.01:
            return
        if d['state'] == 'moving' and 'distance' in d:
            now = time.time()
            if now - self.last_progress >= 5.0:
                self.last_progress = now
                self.say(f"  {d['distance']:.1f} m to go")
        elif d['state'] in ('recovering', 'waiting'):
            self.say('  ' + d.get('message', 'recovering'))
        elif d['state'] in ('succeeded', 'failed', 'canceled'):
            self.nav_result = d
            self.nav_event.set()

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        q = t.transform.rotation
        return t.transform.translation.x, t.transform.translation.y, quaternion_to_yaw(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------ parsing
    def parse(self, text):
        rooms = self.smap.room_names()
        if self.mode != 'keyword' and (self.mode == 'llm' or self.llm_ok):
            t0 = time.time()
            try:
                reply = self.llm.generate(build_prompt(text, rooms, self.smap.classes()))
                steps, problems = validate_llm_steps(extract_json(reply), rooms)
                self.get_logger().info(f'LLM ({time.time() - t0:.1f} s): {reply.strip()[:300]}')
                if problems:
                    self.get_logger().warn('LLM reply problems: ' + '; '.join(problems))
                if steps:
                    return steps, 'llm'
                why = 'unusable LLM reply'
            except RuntimeError as e:
                why = str(e)
            if self.mode == 'llm':
                self.say(f'LLM failed: {why}')
                return [], 'llm'
            self.say(f'LLM failed ({why}); falling back to keyword mode')
        return keyword_parse(text, rooms), 'keyword'

    # ------------------------------------------------------------------ execution
    def _worker(self):
        while rclpy.ok():
            try:
                text = self.commands.get(timeout=0.5)
            except queue.Empty:
                continue
            self.abort.clear()
            self.say(f'> "{text}"')
            steps, how = self.parse(text)
            if not steps:
                self.say("Sorry, I didn't understand that. Try: go to the bed / take me to the kitchen / "
                         'how many chairs are there / what do you see')
                continue
            self.say(f'[{how}] plan: ' + ' -> '.join(describe_step(s) for s in steps))
            for s in steps:
                if self.abort.is_set():
                    break
                if not self.execute(s):
                    if len(steps) > 1:
                        self.say('stopping the remaining steps')
                    break

    def wait_until_ready(self, timeout=90.0):
        """Right after launch the semantic map and the localisation may not be there yet."""
        t0 = time.time()
        told = False
        while time.time() - t0 < timeout and not self.abort.is_set():
            if self.smap.rooms or self.smap.objects:
                if self.robot_pose() is not None:
                    return True
            if not told and time.time() - t0 > 2.0:
                self.say('still starting up (waiting for the semantic map and localisation)...')
                told = True
            time.sleep(0.5)
        return bool(self.smap.rooms or self.smap.objects)

    def execute(self, s):
        a = s['action']
        if a == 'stop':
            self._stop()
            return False
        if a != 'describe':
            self.wait_until_ready()
        if a == 'describe':
            self.describe()
            return True
        if a == 'goto_room':
            room = self.smap.rooms.get(s['room'])
            if room is None:
                self.say(f"I don't know a room called {s['room']}")
                return False
            x, y = room.waypoint
            pose = self.robot_pose()
            yaw = math.atan2(y - pose[1], x - pose[0]) if pose else 0.0
            return self.go(x, y, yaw, f"the {room.name}")

        pose = self.robot_pose()
        best, cands = self.smap.query(s['target'], room=s['room'], near=s['near'], exclude=s['exclude'],
                                      modifier=s['modifier'], robot_xy=pose[:2] if pose else None)
        where = (f" in the {s['room']}" if s['room'] else '') + (f" near the {s['near']}" if s['near'] else '')
        if a == 'count':
            self.say(f"I know of {len(cands)} {s['target']}{'' if len(cands) == 1 else 's'}{where}.")
            return True
        if a == 'list':
            if not cands:
                self.say(f"I haven't seen any {s['target']}{where} yet.")
            for c in cands:
                self.say(f"  {c['class']} #{c['id']} at ({c['x']:.2f}, {c['y']:.2f})"
                         + (f" in the {c['room']}" if c['room'] else '') + f" [{c['source']}]")
            return True
        if best is None:
            seen = ', '.join(self.smap.classes()) or 'nothing yet'
            self.say(f"I haven't seen a {s['target']}{where} yet. On my map: {seen}.")
            return False
        if pose is None:
            self.say('I do not know where I am yet (no map -> base TF). Is localisation running?')
            return False
        order = [best] + [c for c in cands if c is not best]
        for i, obj in enumerate(order[:self.max_candidates]):
            if i:
                self.say(f"trying the next {s['target']}")
            pose = self.robot_pose() or pose
            avoid = [(o['x'], o['y'], OBJECT_RADIUS.get(o['class'], 0.3)) for o in self.smap.confirmed()
                     if o is not obj and o.get('id') != obj.get('id')]
            gx, gy, gyaw = choose_approach(obj['x'], obj['y'], pose[0], pose[1], self.standoff,
                                           grid=self.grid, avoid=avoid)
            label = f"the {obj['class']}" + (f" in the {obj['room']}" if obj['room'] else '') + \
                    f" at ({obj['x']:.2f}, {obj['y']:.2f})"
            if self.go(gx, gy, gyaw, label):
                return True
            if self.abort.is_set():
                return False
        return False

    def go(self, x, y, yaw, label):
        self.nav_event.clear()
        self.nav_result = None
        self.waiting_for = (x, y)
        m = PoseStamped()
        m.header.frame_id = self.map_frame
        m.header.stamp = Time().to_msg()       # 0 = latest transform (see nav2_goal_sender)
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        q = yaw_to_quaternion(yaw)
        m.pose.orientation.z, m.pose.orientation.w = q[2], q[3]
        self.say(f'going to {label}: goal ({x:.2f}, {y:.2f})')
        self.goal_pub.publish(m)
        t0 = time.time()
        while not self.nav_event.wait(0.2):
            if self.abort.is_set():
                self.waiting_for = None
                return False
            if time.time() - t0 > self.goal_timeout:
                self.cancel_pub.publish(Empty())
                self.say(f'gave up on {label} after {self.goal_timeout:.0f} s')
                self.waiting_for = None
                return False
        self.waiting_for = None
        r = self.nav_result or {}
        if r.get('state') == 'succeeded':
            self.say(f'arrived at {label}')
            return True
        if r.get('state') == 'canceled':
            self.say(f'stopped before reaching {label}')
        else:
            self.say(f"could not reach {label}: {r.get('message') or r.get('state')}")
        return False

    def describe(self):
        objs = self.smap.confirmed()
        if not objs:
            self.say('My map is empty so far. Drive around (or wait) so the camera can see things.')
            return
        by_room = {}
        for o in objs:
            by_room.setdefault(o['room'] or 'outside the rooms', []).append(o)
        self.say(f'{len(objs)} objects on my map:')
        for room, items in sorted(by_room.items()):
            counts = {}
            for o in items:
                counts[o['class']] = counts.get(o['class'], 0) + 1
            self.say(f'  {room}: ' + ', '.join(f'{n} {c}' + ('s' if n > 1 else '') for c, n in sorted(counts.items())))

    def _stop(self):
        self.abort.set()
        while not self.commands.empty():
            try:
                self.commands.get_nowait()
            except queue.Empty:
                break
        self.cancel_pub.publish(Empty())
        self.say('stopped')


def main(args=None):
    rclpy.init(args=args)
    node = LlmCommandNode()
    ex = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
