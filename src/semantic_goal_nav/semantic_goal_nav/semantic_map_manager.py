#!/usr/bin/env python3
"""③ Semantic Map Manager node.

Keeps "object label -> map position" and the room polygons.

In:  /detected_objects (from the YOLO node), /scan, /camera/camera_info, TF,
     /map (for the line-of-sight check), optional depth image.
Out: /semantic_map            std_msgs/String JSON (latched): {"objects": [...], "rooms": {...}}
     /semantic_map/markers    visualization_msgs/MarkerArray for RViz

How a box becomes a map point (see projection.py): the LiDAR scan taken at the same moment
is transformed through TF2 into the camera optical frame and projected into the image; the
hits inside the box are the object. With depth_topic set, the depth image is used instead.
The point is then transformed to the map frame at the image timestamp, checked against
the occupancy grid (clear line of sight from the robot, so nothing is mapped "through" a
wall) and merged into the map with a running mean.
"""
import json
import math
import os
from collections import deque

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.duration import Duration as RclDuration
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import OccupancyGrid, apply_transform, make_transform
from .projection import Intrinsics, locate_with_depth, locate_with_lidar, scan_to_points
from .semantic_map import SemanticMap, load_known_objects, load_rooms

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def tf_matrix(t):
    tr, q = t.transform.translation, t.transform.rotation
    return make_transform((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))


class SemanticMapManager(Node):
    def __init__(self):
        super().__init__('semantic_map_manager')
        dyn = ParameterDescriptor(dynamic_typing=True)   # accept 1 as well as 1.0 from YAML/launch
        p = lambda name, value: self.declare_parameter(name, value, dyn)  # noqa: E731
        p('map_frame', 'map')
        p('base_frame', 'base_footprint')
        p('scan_topic', '/scan')
        p('camera_info_topic', '/camera/camera_info')
        p('depth_topic', '')               # e.g. /camera/depth/image_raw (32FC1 metres or 16UC1 mm)
        p('rooms_file', '')
        p('known_objects_file', '')
        p('use_known_objects', True)
        p('save_file', '~/.ros/semantic_map.json')
        p('load_saved', True)
        p('min_confidence', 0.45)
        p('min_observations', 3)
        p('merge_radius', 0.6)
        p('max_object_range', 5.0)
        p('check_line_of_sight', True)
        p('scan_time_tolerance', 0.25)
        p('camera_hfov', 1.085595)         # used only until camera_info arrives
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.map_frame = g('map_frame')
        self.base_frame = g('base_frame')
        self.min_conf = float(g('min_confidence'))
        self.max_range = float(g('max_object_range'))
        self.check_los = bool(g('check_line_of_sight'))
        self.scan_tol_ns = int(float(g('scan_time_tolerance')) * 1e9)
        self.save_file = os.path.expanduser(g('save_file'))
        self.hfov = float(g('camera_hfov'))

        rooms = []
        if g('rooms_file'):
            with open(g('rooms_file')) as f:
                rooms = load_rooms(yaml.safe_load(f))
        self.smap = SemanticMap(rooms, merge_radius=float(g('merge_radius')),
                                min_observations=int(g('min_observations')),
                                clock=lambda: self.get_clock().now().nanoseconds * 1e-9)
        n_known = 0
        if g('use_known_objects') and g('known_objects_file'):
            with open(g('known_objects_file')) as f:
                n_known = load_known_objects(self.smap, yaml.safe_load(f))
        n_saved = self.smap.load(self.save_file) if g('load_saved') else 0
        self.get_logger().info(f'{len(rooms)} rooms, {n_known} known objects, '
                               f'{n_saved} saved detections from {self.save_file}')

        self.tf_buffer = Buffer(cache_time=RclDuration(seconds=15.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.scans = deque(maxlen=40)
        self.intr = None
        self.depth = None
        self.grid = None
        self.dirty = True
        self.stats = {'detections': 0, 'placed': 0, 'no_lidar': 0, 'no_tf': 0, 'far': 0, 'blocked': 0}

        self.create_subscription(String, '/detected_objects', self._on_detections, 10)
        self.create_subscription(LaserScan, g('scan_topic'), self.scans.append, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._on_info, qos_profile_sensor_data)
        if g('depth_topic'):
            self.create_subscription(Image, g('depth_topic'), self._on_depth, qos_profile_sensor_data)
        self.create_subscription(OccupancyGridMsg, '/map', self._on_map, LATCHED)

        self.map_pub = self.create_publisher(String, '/semantic_map', LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_map/markers', LATCHED)
        self.create_timer(1.0, self._publish)
        self.create_timer(10.0, self._save)
        self.create_timer(30.0, self._report)

    # ------------------------------------------------------------------ inputs
    def _on_info(self, msg):
        if self.intr is None:
            self.get_logger().info(f'camera intrinsics: fx={msg.k[0]:.1f} cx={msg.k[2]:.1f} '
                                   f'{msg.width}x{msg.height}')
        self.intr = Intrinsics.from_camera_info(msg)

    def _on_depth(self, msg):
        self.depth = msg

    def _on_map(self, msg):
        self.grid = OccupancyGrid.from_msg(msg)

    def _nearest_scan(self, t_ns):
        best, best_dt = None, None
        for s in self.scans:
            dt = abs(stamp_ns(s.header.stamp) - t_ns)
            if best_dt is None or dt < best_dt:
                best, best_dt = s, dt
        if best is None or best_dt > self.scan_tol_ns:
            return None
        return best

    def _lookup(self, target, source, t_ns):
        """Transform at time t (falls back to the latest one if t is not in the buffer)."""
        try:
            return tf_matrix(self.tf_buffer.lookup_transform(
                target, source, Time(nanoseconds=t_ns), timeout=RclDuration(seconds=0.1)))
        except TransformException:
            try:
                return tf_matrix(self.tf_buffer.lookup_transform(target, source, Time()))
            except TransformException:
                return None

    # ------------------------------------------------------------------ detections
    def _on_detections(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        dets = [x for x in d.get('detections', []) if x.get('confidence', 0) >= self.min_conf]
        if not dets:
            return
        self.stats['detections'] += len(dets)
        t_ns = d['stamp']['sec'] * 1_000_000_000 + d['stamp']['nanosec']
        cam_frame = d.get('frame_id') or 'camera_rgb_optical_frame'
        intr = self.intr or Intrinsics.from_hfov(self.hfov, d.get('width', 640), d.get('height', 480))

        map_T_cam = self._lookup(self.map_frame, cam_frame, t_ns)
        map_T_base = self._lookup(self.map_frame, self.base_frame, t_ns)
        if map_T_cam is None or map_T_base is None:
            self.stats['no_tf'] += len(dets)
            return
        robot_xy = map_T_base[:2, 3]

        depth_img = self._depth_array() if self.depth is not None else None
        pts_cam = None
        if depth_img is None:
            scan = self._nearest_scan(t_ns)
            if scan is None:
                self.stats['no_lidar'] += len(dets)
                return
            cam_T_scan = self._lookup(cam_frame, scan.header.frame_id, t_ns)
            if cam_T_scan is None:
                self.stats['no_tf'] += len(dets)
                return
            pts = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                 scan.range_min, scan.range_max)
            pts_cam = apply_transform(cam_T_scan, pts)

        for det in dets:
            if depth_img is not None:
                p_cam = locate_with_depth(det['bbox'], depth_img, intr, max_depth=self.max_range)
            else:
                p_cam = locate_with_lidar(det['bbox'], pts_cam, intr, max_range=self.max_range)
            if p_cam is None:
                self.stats['no_lidar'] += 1
                continue
            p_map = apply_transform(map_T_cam, p_cam)[0]
            x, y = float(p_map[0]), float(p_map[1])
            if math.hypot(x - robot_xy[0], y - robot_xy[1]) > self.max_range:
                self.stats['far'] += 1
                continue
            if self.check_los and self.grid is not None and \
                    not self.grid.line_of_sight(robot_xy[0], robot_xy[1], x, y, ignore_last=0.4):
                self.stats['blocked'] += 1
                continue
            entry, new = self.smap.add_observation(det['class'], x, y, det['confidence'])
            self.stats['placed'] += 1
            self.dirty = True
            if new:
                self.get_logger().info(f"new {det['class']} at ({x:.2f}, {y:.2f})"
                                       + (f" in the {entry['room']}" if entry['room'] else ''))
            elif entry['observations'] == self.smap.min_observations:
                self.get_logger().info(f"confirmed {entry['class']} #{entry['id']} at "
                                       f"({entry['x']:.2f}, {entry['y']:.2f})")

    def _depth_array(self):
        m = self.depth
        if m.encoding == '32FC1':
            a = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.step // 4)[:, :m.width]
            return a.astype(float)
        if m.encoding in ('16UC1', 'mono16'):
            a = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
            return a.astype(float) / 1000.0
        self.get_logger().warn(f'depth encoding {m.encoding} not supported, using LiDAR',
                               throttle_duration_sec=30.0)
        return None

    # ------------------------------------------------------------------ outputs
    def _publish(self):
        if not self.dirty:
            return
        self.dirty = False
        self.map_pub.publish(String(data=self.smap.to_json()))
        self.marker_pub.publish(self._markers())

    def _markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        now = self.get_clock().now().to_msg()
        for e in self.smap.objects:
            confirmed = e['observations'] >= self.smap.min_observations
            known = e['source'] == 'known'
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns, m.id, m.type = 'objects', e['id'], Marker.CYLINDER
            m.pose.position.x, m.pose.position.y, m.pose.position.z = e['x'], e['y'], 0.15
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.25
            m.scale.z = 0.3
            m.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.5) if known else \
                (ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.9) if confirmed else ColorRGBA(r=1.0, g=0.7, b=0.0, a=0.6))
            arr.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type = 'labels', e['id'], Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = e['x'], e['y'], 0.55
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = e['class'] + ('' if known else f" ({e['observations']})")
            arr.markers.append(t)
        for i, room in enumerate(self.smap.rooms.values()):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns, m.id, m.type = 'rooms', i, Marker.LINE_STRIP
            m.pose.orientation.w = 1.0
            m.scale.x = 0.04
            m.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.8)
            m.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in room.polygon + room.polygon[:1]]
            arr.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type = 'room_names', i, Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y, t.pose.position.z = room.waypoint[0], room.waypoint[1], 0.9
            t.pose.orientation.w = 1.0
            t.scale.z = 0.35
            t.color = ColorRGBA(r=0.4, g=0.7, b=1.0, a=1.0)
            t.text = room.name
            arr.markers.append(t)
        for mk in arr.markers:
            mk.lifetime = Duration()
        return arr

    def _save(self):
        try:
            self.smap.save(self.save_file)
        except OSError as e:
            self.get_logger().warn(f'could not save {self.save_file}: {e}', throttle_duration_sec=60.0)

    def _report(self):
        s = self.stats
        if s['detections']:
            self.get_logger().info(
                f"detections {s['detections']}: placed {s['placed']}, no LiDAR hit {s['no_lidar']}, "
                f"no TF {s['no_tf']}, too far {s['far']}, behind a wall {s['blocked']} | "
                f"map: {len(self.smap.confirmed())} confirmed objects")


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
