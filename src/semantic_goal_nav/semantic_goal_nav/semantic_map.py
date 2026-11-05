"""The semantic map: object label -> map position, plus room polygons (pure Python).

Each object entry is a dict:
    {'id': 7, 'class': 'chair', 'x': 6.25, 'y': 0.22, 'confidence': 0.71,
     'observations': 12, 'last_seen': 1723.4, 'room': 'kitchen', 'source': 'detected'}

source = 'detected' (built from YOLO + LiDAR/depth) or 'known' (from known_objects.yaml,
the objects placed in the Gazebo world). A detection of the same class near an entry
updates it with a running mean, so positions settle as sightings add up instead of
jumping with every noisy frame.
"""
import json
import math
import os
import time

from .geometry import grid_points_in_polygon, point_in_polygon, polygon_centroid
from .vocab import normalize_class, normalize_room, register_room_aliases

ORDINALS = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4}

# Big furniture: LiDAR hits land anywhere along its front, so sightings of the same object
# are further apart than for a chair. Same-class sightings closer than this are merged.
MERGE_RADIUS = {'bed': 1.5, 'couch': 1.2, 'dining table': 1.2, 'desk': 1.0, 'wardrobe': 1.0,
                'coffee table': 0.9, 'refrigerator': 0.8, 'cabinet': 0.9, 'tv': 0.8}


class Room:
    def __init__(self, name, polygon, waypoint=None, aliases=None):
        self.name = name
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.waypoint = tuple(waypoint) if waypoint else polygon_centroid(self.polygon)
        self.aliases = list(aliases or [])
        register_room_aliases({a: name for a in self.aliases})

    def contains(self, x, y):
        return point_in_polygon(x, y, self.polygon)

    def candidate_points(self, spacing=0.5):
        return grid_points_in_polygon(self.polygon, spacing)


class SemanticMap:
    def __init__(self, rooms=None, merge_radius=0.6, max_weight=30, min_observations=3,
                 clock=time.time):
        self.rooms = {r.name: r for r in (rooms or [])}
        self.merge_radius = merge_radius
        self.max_weight = max_weight          # caps the running mean so moved objects can still update
        self.min_observations = min_observations
        self.clock = clock
        self.objects = []
        self._next_id = 1

    # ------------------------------------------------------------ rooms
    def room_of(self, x, y):
        for r in self.rooms.values():
            if r.contains(x, y):
                return r.name
        return None

    def room_names(self):
        return list(self.rooms)

    # ------------------------------------------------------------ objects
    def add_observation(self, cls, x, y, confidence, now=None):
        """Merge one detection into the map. Returns (entry, is_new)."""
        cls = normalize_class(cls) or cls
        now = self.clock() if now is None else now
        best, best_d = None, max(self.merge_radius, MERGE_RADIUS.get(cls, 0.0))
        for e in self.objects:
            if e['class'] != cls:
                continue
            d = math.hypot(e['x'] - x, e['y'] - y)
            if d <= best_d:
                best, best_d = e, d
        if best is not None:
            n = min(best['observations'], self.max_weight)
            best['x'] = (best['x'] * n + x) / (n + 1)
            best['y'] = (best['y'] * n + y) / (n + 1)
            best['observations'] += 1
            best['confidence'] = max(best['confidence'], round(float(confidence), 3))
            best['last_seen'] = now
            best['room'] = self.room_of(best['x'], best['y'])
            if best['source'] == 'known':
                best['source'] = 'known+detected'
            return best, False
        e = {'id': self._next_id, 'class': cls, 'x': float(x), 'y': float(y),
             'confidence': round(float(confidence), 3), 'observations': 1,
             'last_seen': now, 'room': self.room_of(x, y), 'source': 'detected'}
        self._next_id += 1
        self.objects.append(e)
        return e, True

    def add_known(self, cls, x, y, name=None):
        """An object we know is there (e.g. from the world file); counts as confirmed."""
        c = normalize_class(cls) or cls
        e = {'id': self._next_id, 'class': c, 'x': float(x), 'y': float(y), 'confidence': 1.0,
             'observations': self.min_observations, 'last_seen': 0.0,
             'room': self.room_of(x, y), 'source': 'known'}
        if name:
            e['name'] = name
        self._next_id += 1
        self.objects.append(e)
        return e

    def confirmed(self):
        """Entries seen often enough to navigate to (single-frame false positives stay out)."""
        return [e for e in self.objects if e['observations'] >= self.min_observations]

    def classes(self):
        return sorted({e['class'] for e in self.confirmed()})

    def query(self, target, room=None, near=None, exclude=None, modifier=None,
              robot_xy=None, near_radius=1.5):
        """Objects of class `target`, filtered and sorted. Returns (best or None, candidates).

        room     only objects inside that room
        near     only objects within near_radius of an object of that class
        exclude  drop objects within near_radius of an object of that class
        modifier nearest (default) / farthest / first / second / ... (ordered by distance)
        """
        target = normalize_class(target) or target
        objs = self.confirmed()
        cands = [e for e in objs if e['class'] == target]
        if room:
            cands = [e for e in cands if e['room'] == room]
        if near:
            refs = [e for e in objs if e['class'] == normalize_class(near)]
            cands = [c for c in cands if any(_dist(c, r) <= near_radius for r in refs)]
            if refs and cands:
                cands.sort(key=lambda c: min(_dist(c, r) for r in refs))
        if exclude:
            bad = [e for e in objs if e['class'] == normalize_class(exclude)]
            cands = [c for c in cands if all(_dist(c, b) > near_radius for b in bad)]
        if not cands:
            return None, []
        if robot_xy is not None and not near:
            rx, ry = robot_xy
            cands.sort(key=lambda e: math.hypot(e['x'] - rx, e['y'] - ry))
        if modifier == 'farthest':
            return cands[-1], cands
        if modifier in ORDINALS:
            i = ORDINALS[modifier]
            return (cands[i] if i < len(cands) else None), cands
        return cands[0], cands

    # ------------------------------------------------------------ persistence
    def to_dict(self):
        return {'objects': self.objects,
                'rooms': {n: {'polygon': r.polygon, 'waypoint': list(r.waypoint), 'aliases': r.aliases}
                          for n, r in self.rooms.items()}}

    def to_json(self):
        return json.dumps(self.to_dict())

    def save(self, path):
        """Save detected objects only (known ones are reloaded from their file)."""
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump([e for e in self.objects if e['source'] != 'known'], f, indent=1)
        os.replace(tmp, path)

    def load(self, path):
        """Merge previously saved detections. Returns how many entries were read."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return 0
        if not isinstance(data, list):
            return 0
        n = 0
        for e in data:
            try:
                cls, x, y = e['class'], float(e['x']), float(e['y'])
            except (KeyError, TypeError, ValueError):
                continue
            entry, new = self.add_observation(cls, x, y, e.get('confidence', 0.5), now=e.get('last_seen', 0))
            entry['observations'] = max(entry['observations'], int(e.get('observations', 1)))
            n += 1
        return n

    @classmethod
    def from_dict(cls, d, **kw):
        """Rebuild a read-only view from to_dict() output (what /semantic_map carries)."""
        rooms = [Room(n, r['polygon'], r.get('waypoint'), r.get('aliases'))
                 for n, r in (d.get('rooms') or {}).items()]
        m = cls(rooms, **kw)
        m.objects = list(d.get('objects') or [])
        m._next_id = max([e.get('id', 0) for e in m.objects] + [0]) + 1
        return m

    def normalize_room(self, name):
        return normalize_room(name, self.room_names())


def _dist(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def load_rooms(data):
    """rooms.yaml content -> [Room].
    Format: {rooms: {name: {polygon: [[x,y],...], waypoint: [x,y], aliases: [other names]}}}"""
    rooms = []
    for name, r in ((data or {}).get('rooms') or {}).items():
        rooms.append(Room(name, r['polygon'], r.get('waypoint'), r.get('aliases')))
    return rooms


def load_known_objects(smap, data):
    """known_objects.yaml content -> entries in smap. Format: {objects: [{class, x, y, name}]}"""
    n = 0
    for o in (data or {}).get('objects') or []:
        if normalize_class(o.get('class')):
            smap.add_known(o['class'], o['x'], o['y'], o.get('name'))
            n += 1
    return n
