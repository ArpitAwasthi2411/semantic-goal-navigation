"""Geometry helpers (pure Python + numpy, no ROS): polygons, quaternions, transforms,
approach points and a small occupancy-grid wrapper used for goal checks and ray tracing."""
import math

import numpy as np


# ---------------------------------------------------------------- polygons / rooms

def point_in_polygon(x, y, polygon):
    """Even-odd rule. polygon: list of (x, y) in order (clockwise or anticlockwise)."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_centroid(polygon):
    """Area centroid of a simple polygon (falls back to the vertex mean if degenerate)."""
    a = cx = cy = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-9:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def grid_points_in_polygon(polygon, spacing=0.5):
    """Regular grid of candidate points inside a room (used to pick a free room goal)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    pts = []
    x = min(xs) + spacing / 2
    while x < max(xs):
        y = min(ys) + spacing / 2
        while y < max(ys):
            if point_in_polygon(x, y, polygon):
                pts.append((round(x, 3), round(y, 3)))
            y += spacing
        x += spacing
    return pts


# ---------------------------------------------------------------- rotations / transforms

def yaw_to_quaternion(yaw):
    """(x, y, z, w) for a rotation about z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_matrix(x, y, z, w):
    """3x3 rotation matrix of a unit quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_transform(translation, quaternion):
    """4x4 homogeneous matrix from (tx, ty, tz) and (qx, qy, qz, qw)."""
    m = np.eye(4)
    m[:3, :3] = quaternion_matrix(*quaternion)
    m[:3, 3] = translation
    return m


def apply_transform(m, points):
    """Apply a 4x4 transform to an (N, 3) array of points."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    return p @ m[:3, :3].T + m[:3, 3]


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------- goals

def approach_point(obj_x, obj_y, robot_x, robot_y, standoff):
    """A point `standoff` metres from the object, on the side the robot is coming from,
    plus the yaw that faces the object. Driving to the object's own centre is impossible
    (it is an obstacle), so Nav2 always gets this point instead."""
    dx, dy = robot_x - obj_x, robot_y - obj_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        dx, dy, d = 1.0, 0.0, 1.0
    s = min(standoff, d)
    gx, gy = obj_x + dx / d * s, obj_y + dy / d * s
    return gx, gy, math.atan2(obj_y - gy, obj_x - gx)


class OccupancyGrid:
    """Minimal occupancy grid (nav_msgs/OccupancyGrid layout: row-major, origin at cell (0,0))."""

    def __init__(self, width, height, resolution, origin_x, origin_y, data):
        self.w, self.h, self.res = int(width), int(height), float(resolution)
        self.ox, self.oy = float(origin_x), float(origin_y)
        self.data = np.asarray(data, dtype=np.int16).reshape(self.h, self.w)
        self._clearance = None

    @classmethod
    def from_msg(cls, msg):
        i = msg.info
        return cls(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y, msg.data)

    @classmethod
    def from_pgm(cls, image, resolution, origin_x, origin_y, occupied_thresh=0.65, free_thresh=0.196):
        """From a map_server style greyscale image (row 0 = top), negate = 0."""
        img = np.asarray(image, dtype=float)
        p = (255.0 - img) / 255.0
        data = np.full(img.shape, -1, dtype=np.int16)
        data[p > occupied_thresh] = 100
        data[p < free_thresh] = 0
        return cls(img.shape[1], img.shape[0], resolution, origin_x, origin_y, np.flipud(data).ravel())

    def cell(self, x, y):
        return int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))

    def world(self, cx, cy):
        return self.ox + (cx + 0.5) * self.res, self.oy + (cy + 0.5) * self.res

    def inside(self, cx, cy):
        return 0 <= cx < self.w and 0 <= cy < self.h

    def value(self, x, y):
        cx, cy = self.cell(x, y)
        return int(self.data[cy, cx]) if self.inside(cx, cy) else -1

    def is_free(self, x, y):
        return 0 <= self.value(x, y) < 50

    def clearance_map(self):
        """Distance (m) from every cell to the nearest occupied/unknown cell."""
        if self._clearance is None:
            blocked = (self.data >= 50) | (self.data < 0)
            try:
                from scipy import ndimage
                self._clearance = ndimage.distance_transform_edt(~blocked) * self.res
            except ImportError:  # small brute-force fallback (slow but correct)
                self._clearance = _edt_fallback(blocked) * self.res
        return self._clearance

    def clearance(self, x, y):
        cx, cy = self.cell(x, y)
        if not self.inside(cx, cy):
            return 0.0
        return float(self.clearance_map()[cy, cx])

    def nearest_clear(self, x, y, min_clearance, max_search=1.5):
        """Closest point to (x, y) whose clearance is >= min_clearance, or None."""
        if self.clearance(x, y) >= min_clearance:
            return x, y
        cmap = self.clearance_map()
        cx, cy = self.cell(x, y)
        r = int(math.ceil(max_search / self.res))
        x0, x1 = max(cx - r, 0), min(cx + r + 1, self.w)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, self.h)
        if x0 >= x1 or y0 >= y1:
            return None
        sub = cmap[y0:y1, x0:x1]
        yy, xx = np.nonzero(sub >= min_clearance)
        if len(xx) == 0:
            return None
        d2 = (xx + x0 - cx) ** 2 + (yy + y0 - cy) ** 2
        k = int(np.argmin(d2))
        if d2[k] > r * r:
            return None
        return self.world(int(xx[k] + x0), int(yy[k] + y0))

    def line_of_sight(self, x0, y0, x1, y1, ignore_last=0.3):
        """True when no occupied cell lies on the segment, except within `ignore_last`
        metres of the end point (the object itself is an obstacle)."""
        d = math.hypot(x1 - x0, y1 - y0)
        if d <= ignore_last:
            return True
        steps = int((d - ignore_last) / (self.res * 0.5)) + 1
        for i in range(steps):
            t = i * self.res * 0.5 / d
            v = self.value(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if v >= 50:
                return False
        return True


def reach_field(grid, x, y, min_clearance):
    """Driving distance (m) from (x, y) to every cell, through cells with clearance >=
    min_clearance (8-connected Dijkstra-like BFS). inf where unreachable."""
    import heapq
    clr = grid.clearance_map()
    ok = clr >= min_clearance
    dist = np.full(ok.shape, np.inf)
    cx, cy = grid.cell(x, y)
    if not grid.inside(cx, cy):
        return dist
    if not ok[cy, cx]:          # start may sit in a tight spot: let it out
        ok = ok.copy()
        ok[max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5] = True
    dist[cy, cx] = 0.0
    heap = [(0.0, cx, cy)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[j, i]:
            continue
        for di, dj, c in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.w and 0 <= nj < grid.h and ok[nj, ni]:
                nd = d + c * grid.res
                if nd < dist[nj, ni]:
                    dist[nj, ni] = nd
                    heapq.heappush(heap, (nd, ni, nj))
    return dist


def choose_approach(obj_x, obj_y, robot_x, robot_y, standoff, grid=None, avoid=(), min_clearance=0.3,
                    field=None):
    """Where to stop next to an object: a point about `standoff` from it, on free floor, not
    on top of other furniture (e.g. under the table next to a chair), and with the shortest
    drive from the robot. avoid: [(x, y, radius)] of other known objects.
    Returns (x, y, yaw facing the object). Falls back to approach_point() without a grid."""
    if grid is None:
        return approach_point(obj_x, obj_y, robot_x, robot_y, standoff)
    if field is None:
        field = reach_field(grid, robot_x, robot_y, min_clearance * 0.8)
    best = None
    for extra in (0.0, 0.2, 0.4, 0.7):
        r = standoff + extra
        for k in range(24):
            a = k * math.pi / 12
            px, py = obj_x + r * math.cos(a), obj_y + r * math.sin(a)
            if grid.clearance(px, py) < min_clearance:
                continue
            if any(math.hypot(px - ax, py - ay) < ar for ax, ay, ar in avoid):
                continue
            cx, cy = grid.cell(px, py)
            if not grid.inside(cx, cy) or not np.isfinite(field[cy, cx]):
                continue
            score = field[cy, cx] + 2.0 * extra
            if best is None or score < best[0]:
                best = (score, px, py)
        if best is not None:
            break
    if best is None:
        return approach_point(obj_x, obj_y, robot_x, robot_y, standoff)
    _, px, py = best
    return px, py, math.atan2(obj_y - py, obj_x - px)


def _edt_fallback(blocked):
    h, w = blocked.shape
    out = np.full((h, w), np.inf)
    ys, xs = np.nonzero(blocked)
    if len(xs) == 0:
        return np.full((h, w), 1e6)
    gy, gx = np.mgrid[0:h, 0:w]
    for i in range(0, len(xs), 512):
        bx, by = xs[i:i + 512], ys[i:i + 512]
        d = np.sqrt((gx[..., None] - bx) ** 2 + (gy[..., None] - by) ** 2).min(axis=-1)
        out = np.minimum(out, d)
    return out
