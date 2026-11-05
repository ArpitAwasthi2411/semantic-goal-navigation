#!/usr/bin/env python3
"""② YOLOv8n Detector node.

Subscribes to the RGB camera, runs YOLOv8n and publishes
  /detected_objects        std_msgs/String (JSON, see below)
  /yolo/annotated_image    sensor_msgs/Image (boxes drawn, for RViz / rqt_image_view)

/detected_objects JSON:
  {"stamp": {"sec": 12, "nanosec": 5000}, "frame_id": "camera_rgb_optical_frame",
   "width": 640, "height": 480,
   "detections": [{"class": "chair", "raw_class": "chair", "confidence": 0.81,
                   "bbox": [x1, y1, x2, y2]}]}

Only classes the semantic map knows (vocab.KNOWN_CLASSES) are published. Inference runs
on a timer at `max_rate_hz` on the newest frame, so the node never falls behind the camera.
`half:=true` uses FP16 (needs a CUDA GPU; ignored on CPU). `model` can be yolov8n.pt (COCO,
downloaded by Ultralytics on first use) or the path to your own trained weights.
"""
import json
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .vocab import normalize_class


def image_to_bgr(msg):
    """sensor_msgs/Image -> BGR numpy array without cv_bridge (rgb8 / bgr8 / mono8)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('rgb8', 'bgr8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
    if msg.encoding in ('mono8', '8UC1'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if msg.encoding in ('rgba8', 'bgra8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_RGBA2BGR if msg.encoding == 'rgba8' else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(img, code)
    raise ValueError(f'unsupported image encoding {msg.encoding}')


def bgr_to_image(img, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        dyn = ParameterDescriptor(dynamic_typing=True)   # accept 1 as well as 1.0 from YAML/launch
        p = lambda name, value: self.declare_parameter(name, value, dyn)  # noqa: E731
        p('model', 'yolov8n.pt')
        p('image_topic', '/camera/image_raw')
        p('confidence', 0.4)
        p('iou', 0.45)
        p('imgsz', 640)
        p('max_rate_hz', 4.0)
        p('device', '')                 # '' = auto (cuda if available), or 'cpu', 'cuda:0'
        p('half', False)                # FP16 inference (CUDA only)
        p('publish_annotated', True)
        p('open_vocab_classes', [''])   # for YOLO-World weights: the words to look for
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.conf = float(g('confidence'))
        self.iou = float(g('iou'))
        self.imgsz = int(g('imgsz'))
        self.publish_annotated = bool(g('publish_annotated'))

        from ultralytics import YOLO
        model_name = g('model')
        self.get_logger().info(f'loading {model_name} ...')
        t0 = time.time()
        self.model = YOLO(model_name)
        words = [w for w in g('open_vocab_classes') if w]
        if words and hasattr(self.model, 'set_classes'):
            self.model.set_classes(words)
        dev = g('device')
        if not dev:
            try:
                import torch
                dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                dev = 'cpu'
        self.device = dev
        self.half = bool(g('half')) and dev.startswith('cuda')
        if g('half') and not self.half:
            self.get_logger().warn('half:=true needs a CUDA GPU; running FP32 on CPU')
        # warm-up so the first real frame is not slow
        self.extra = {'half': True} if self.half else {}    # newer Ultralytics warns on half=False
        self.model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, device=self.device,
                           verbose=False, **self.extra)
        self.get_logger().info(f'model ready on {self.device} in {time.time() - t0:.1f} s '
                               f'({len(self.model.names)} classes, conf >= {self.conf})')

        self.latest = None
        self.det_pub = self.create_publisher(String, '/detected_objects', 10)
        self.img_pub = self.create_publisher(Image, '/yolo/annotated_image', 2)
        self.create_subscription(Image, g('image_topic'), self._on_image, qos_profile_sensor_data)
        self.create_timer(1.0 / max(float(g('max_rate_hz')), 0.1), self._tick)
        self.frames = 0
        self.inference_ms = 0.0
        self.create_timer(10.0, self._report)
        self._warned_skip = set()

    def _on_image(self, msg):
        self.latest = msg

    def _tick(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        try:
            frame = image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=10.0)
            return
        t0 = time.time()
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False, **self.extra)[0]
        ms = (time.time() - t0) * 1000.0
        self.inference_ms = ms if self.frames == 0 else 0.9 * self.inference_ms + 0.1 * ms
        self.frames += 1

        dets = []
        for box in res.boxes:
            raw = self.model.names[int(box.cls[0])]
            cls = normalize_class(raw)
            if cls is None:
                if raw not in self._warned_skip:
                    self._warned_skip.add(raw)
                    self.get_logger().info(f"ignoring class '{raw}' (not a map object)")
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            dets.append({'class': cls, 'raw_class': raw, 'confidence': round(float(box.conf[0]), 3),
                         'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})

        out = {'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
               'frame_id': msg.header.frame_id, 'width': msg.width, 'height': msg.height,
               'detections': dets}
        self.det_pub.publish(String(data=json.dumps(out)))

        if self.publish_annotated and self.img_pub.get_subscription_count() > 0:
            img = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d['bbox'])
                color = _color(d['class'])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{d['class']} {d['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f'{self.inference_ms:.0f} ms', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            self.img_pub.publish(bgr_to_image(img, msg.header))

    def _report(self):
        if self.frames:
            self.get_logger().info(f'{self.frames} frames processed, inference {self.inference_ms:.0f} ms/frame')


def _color(name):
    h = sum(ord(c) * 31 ** i for i, c in enumerate(name)) % 180
    hsv = np.uint8([[[h, 200, 230]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
