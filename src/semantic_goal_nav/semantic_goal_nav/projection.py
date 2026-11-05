"""Where is the detected object? (pure numpy, no ROS)

Two ways to turn a 2D bounding box into a 3D point:

1. LiDAR (default; the TurtleBot3 Waffle Pi has an RGB camera but no depth camera):
   every LiDAR hit is transformed into the camera's optical frame through TF2 and
   projected into the image with the camera intrinsics. The hits whose pixel column
   falls inside the box (its central part) are the hits on the object. Their nearest
   cluster gives the object's position. This uses the real camera and LiDAR mounts
   from TF instead of a fixed offset, so there is no height/offset error.

2. Depth image (when a depth camera is present): the median depth inside the central
   part of the box, back-projected through the intrinsics.

Coordinates: optical frame = x right, y down, z forward (REP 103).
"""
import numpy as np


class Intrinsics:
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    @classmethod
    def from_camera_info(cls, msg):
        k = msg.k
        return cls(k[0], k[4], k[2], k[5], msg.width, msg.height)

    @classmethod
    def from_hfov(cls, hfov_rad, width, height):
        f = (width / 2.0) / np.tan(hfov_rad / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    def project(self, pts_optical):
        """(N,3) optical-frame points -> (u, v) pixel arrays (NaN where z <= 0)."""
        p = np.asarray(pts_optical, dtype=float).reshape(-1, 3)
        z = np.where(p[:, 2] > 1e-6, p[:, 2], np.nan)
        return self.fx * p[:, 0] / z + self.cx, self.fy * p[:, 1] / z + self.cy

    def back_project(self, u, v, depth):
        return np.array([(u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth])


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """LaserScan arrays -> (N,3) points in the scan frame (invalid beams dropped)."""
    r = np.asarray(ranges, dtype=float)
    a = angle_min + np.arange(len(r)) * angle_increment
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    r, a = r[ok], a[ok]
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1)


def shrink_bbox(bbox, keep=0.5):
    """Central part of a box (fraction `keep` of width/height): avoids background at the edges."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * keep / 2.0, (y2 - y1) * keep / 2.0
    return cx - hw, cy - hh, cx + hw, cy + hh


def locate_with_lidar(bbox, pts_optical, intr, keep=0.7, depth_window=0.35, min_hits=1,
                      max_range=6.0):
    """Object point (optical frame) from LiDAR hits inside the box columns, or None.

    pts_optical: (N,3) LiDAR hits already transformed into the camera optical frame.
    The object is in front of whatever is behind it, so the nearest hits in those columns
    belong to it: take the hits within `depth_window` of the nearest one and average them.
    Thin things (chair legs) give only a few hits among many wall hits behind them, so a
    percentile would land on the wall; the nearest hit is the right anchor, and one hit
    is enough."""
    p = np.asarray(pts_optical, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        return None
    u, _ = intr.project(p)
    x1, _, x2, _ = shrink_bbox(bbox, keep)
    sel = np.isfinite(u) & (u >= x1) & (u <= x2) & (p[:, 2] > 0.05)
    hits = p[sel]
    if len(hits) < min_hits:
        return None
    d = np.linalg.norm(hits[:, [0, 2]], axis=1)
    near = float(d.min())
    if near > max_range:
        return None
    obj = hits[(d >= near - 0.05) & (d <= near + depth_window)]
    if len(obj) < min_hits:
        return None
    return obj.mean(axis=0)


def locate_with_depth(bbox, depth, intr, keep=0.4, min_depth=0.1, max_depth=6.0):
    """Object point (optical frame) from a depth image (metres, HxW), or None."""
    x1, y1, x2, y2 = (int(round(v)) for v in shrink_bbox(bbox, keep))
    h, w = depth.shape[:2]
    x1, x2 = max(x1, 0), min(max(x2, x1 + 1), w)
    y1, y2 = max(y1, 0), min(max(y2, y1 + 1), h)
    patch = np.asarray(depth[y1:y2, x1:x2], dtype=float)
    patch = patch[np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)]
    if patch.size < 5:
        return None
    z = float(np.percentile(patch, 40))     # biased to the front surface
    bx1, by1, bx2, by2 = bbox
    return intr.back_project((bx1 + bx2) / 2.0, (by1 + by2) / 2.0, z)
