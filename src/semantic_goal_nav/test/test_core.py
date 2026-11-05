"""Tests for the pure-Python parts: parsing, semantic map, projection, geometry.
Run:  cd src/semantic_goal_nav && python3 -m pytest test -q   (no ROS needed)"""
import json
import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from semantic_goal_nav.command_parser import (build_prompt, extract_json, keyword_parse,  # noqa: E402
                                         validate_llm_steps)
from semantic_goal_nav.geometry import (OccupancyGrid, approach_point, apply_transform,  # noqa: E402
                                   make_transform, point_in_polygon, yaw_to_quaternion)
from semantic_goal_nav.projection import (Intrinsics, locate_with_depth, locate_with_lidar,  # noqa: E402
                                     scan_to_points)
from semantic_goal_nav.semantic_map import SemanticMap, load_known_objects, load_rooms  # noqa: E402
from semantic_goal_nav.vocab import normalize_class, normalize_room  # noqa: E402

PKG = os.path.join(os.path.dirname(__file__), '..')
ROOMS = load_rooms(yaml.safe_load(open(os.path.join(PKG, 'config', 'rooms.yaml'))))
ROOM_NAMES = [r.name for r in ROOMS]


# ------------------------------------------------------------------ vocabulary

@pytest.mark.parametrize('word,canon', [
    ('sofa', 'couch'), ('Fridge', 'refrigerator'), ('television', 'tv'), ('chairs', 'chair'),
    ('the bed', 'bed'), ('dining_table', 'dining table'), ('almirah', 'wardrobe'), ('seating', 'couch'),
    ('spaceship', None), ('', None)])
def test_normalize_class(word, canon):
    assert normalize_class(word) == canon


def test_normalize_room():
    assert normalize_room('Living Room', ROOM_NAMES) == 'living room'
    assert normalize_room('lounge', ROOM_NAMES) == 'living room'
    assert normalize_room('the dining room', ROOM_NAMES) == 'kitchen'
    assert normalize_room('garage', ROOM_NAMES) is None


# ------------------------------------------------------------------ keyword parser

def kp(text):
    return keyword_parse(text, ROOM_NAMES)


def test_keyword_goto_object_and_synonym():
    s = kp('go to the sofa')[0]
    assert (s['action'], s['target'], s['modifier']) == ('goto', 'couch', 'nearest')
    assert kp('take me to the seating')[0]['target'] == 'couch'


def test_keyword_room_and_object_in_room():
    assert kp('take me to the kitchen')[0] == {'action': 'goto_room', 'target': None, 'room': 'kitchen',
                                               'near': None, 'exclude': None, 'modifier': None}
    s = kp('go to the chair in the bedroom')[0]
    assert (s['target'], s['room']) == ('chair', 'bedroom')
    s = kp('go to the tv room')[0]          # "tv room" is a room, not a tv
    assert (s['action'], s['room']) == ('goto_room', 'study')


def test_keyword_qualifiers():
    s = kp('go to the chair near the table')[0]
    assert (s['target'], s['near']) == ('chair', 'dining table')
    s = kp('go to the farthest chair but avoid the person')[0]
    assert (s['target'], s['modifier'], s['exclude']) == ('chair', 'farthest', 'person')
    assert kp('go to the second chair')[0]['modifier'] == 'second'


def test_keyword_questions_and_stop():
    assert kp('how many chairs are there')[0]['action'] == 'count'
    assert kp('where are the chairs')[0]['action'] == 'list'
    assert kp('what do you see')[0]['action'] == 'describe'
    assert kp('stop')[0]['action'] == 'stop'
    assert kp('blah blah') == []


def test_keyword_sequence():
    steps = kp('go to the bed and then go to the fridge')
    assert [s['target'] for s in steps] == ['bed', 'refrigerator']


# ------------------------------------------------------------------ LLM output handling

def test_extract_json_with_chatter_and_fences():
    reply = 'Sure! ```json\n{"steps": [{"action": "goto", "target": "chair"}]}\n``` hope that helps {'
    assert extract_json(reply)['steps'][0]['target'] == 'chair'
    assert extract_json('no json here') is None
    assert extract_json('{"a": "}"}') == {'a': '}'}


def test_validate_llm_steps_normalises_and_rejects():
    data = {'steps': [{'action': 'goto', 'target': 'Sofa', 'room': 'Living Room', 'modifier': None},
                      {'action': 'goto', 'target': 'unicorn'},
                      {'action': 'fly', 'target': 'bed'},
                      {'action': 'goto_room', 'room': 'garage'},
                      {'action': 'goto', 'target': None, 'room': 'kitchen'}]}
    steps, problems = validate_llm_steps(data, ROOM_NAMES)
    assert steps[0] == {'action': 'goto', 'target': 'couch', 'room': 'living room', 'near': None,
                        'exclude': None, 'modifier': 'nearest'}
    assert steps[1]['action'] == 'goto_room' and steps[1]['room'] == 'kitchen'
    assert len(steps) == 2 and len(problems) == 5


def test_validate_accepts_old_schema_and_single_step():
    steps, _ = validate_llm_steps({'action': 'goto', 'target_class': 'bed', 'reference_class': 'tv'}, ROOM_NAMES)
    assert steps[0]['target'] == 'bed' and steps[0]['near'] == 'tv'
    assert validate_llm_steps('garbage', ROOM_NAMES) == ([], ['reply has no "steps" list'])


def test_prompt_lists_rooms_and_seen_objects():
    p = build_prompt('go to the bed', ROOM_NAMES, ['bed', 'chair'])
    assert 'living room' in p and 'Objects currently on the robot\'s map: bed, chair' in p
    assert p.rstrip().endswith('JSON:')


# ------------------------------------------------------------------ semantic map

def make_map():
    return SemanticMap(ROOMS, merge_radius=0.6, min_observations=3, clock=lambda: 0.0)


def test_running_mean_merges_and_confirms():
    m = make_map()
    for x in (1.0, 1.2, 1.1, 0.9):
        e, new = m.add_observation('chair', x, 0.0, 0.8)
    assert len(m.objects) == 1 and e['observations'] == 4
    assert abs(e['x'] - 1.05) < 1e-9
    assert m.confirmed() == [e]
    m.add_observation('chair', 3.0, 0.0, 0.9)                 # far away: a second chair
    assert len(m.objects) == 2 and len(m.confirmed()) == 1     # but not confirmed yet


def test_room_assignment_and_query_filters():
    m = make_map()
    for _ in range(3):
        m.add_observation('chair', 6.2, 0.2, 0.9)      # kitchen
        m.add_observation('chair', -8.2, 1.9, 0.9)     # bedroom
        m.add_observation('dining table', 6.5, 0.9, 0.9)
        m.add_observation('person', -8.0, 1.5, 0.9)
    best, cands = m.query('chair', robot_xy=(0.0, 0.0))
    assert best['room'] == 'kitchen' and len(cands) == 2      # nearest first
    assert m.query('chair', room='bedroom')[0]['x'] == pytest.approx(-8.2)
    assert m.query('chair', near='table')[0]['room'] == 'kitchen'
    assert m.query('chair', exclude='person', robot_xy=(-8, 0))[0]['room'] == 'kitchen'
    assert m.query('chair', modifier='farthest', robot_xy=(0, 0))[0]['room'] == 'bedroom'
    assert m.query('chair', modifier='third', robot_xy=(0, 0))[0] is None
    assert m.query('sofa')[0] is None


def test_known_objects_file_loads_and_detections_refine():
    m = make_map()
    n = load_known_objects(m, yaml.safe_load(open(os.path.join(PKG, 'config', 'known_objects.yaml'))))
    assert n >= 25
    bed = m.query('bed')[0]
    assert bed['room'] == 'bedroom' and bed['source'] == 'known'
    e, new = m.add_observation('bed', bed['x'] + 0.2, bed['y'], 0.7)
    assert not new and e['source'] == 'known+detected'
    assert m.query('refrigerator')[0]['room'] == 'kitchen'
    assert m.query('couch')[0]['room'] == 'living room'


def test_save_load_roundtrip(tmp_path):
    m = make_map()
    m.add_known('bed', 0, 0)
    for _ in range(4):
        m.add_observation('tv', 2.0, 1.0, 0.8)
    f = str(tmp_path / 'sm.json')
    m.save(f)
    saved = json.load(open(f))
    assert [e['class'] for e in saved] == ['tv']             # known objects are not saved
    m2 = make_map()
    assert m2.load(f) == 1 and m2.confirmed()[0]['observations'] == 4
    assert m2.load(str(tmp_path / 'missing.json')) == 0


def test_to_dict_from_dict():
    m = make_map()
    for _ in range(3):
        m.add_observation('bed', -6, 2, 0.9)
    m2 = SemanticMap.from_dict(json.loads(m.to_json()))
    assert m2.query('bed')[0]['room'] == 'bedroom' and set(m2.room_names()) == set(ROOM_NAMES)


# ------------------------------------------------------------------ rooms config sanity

def test_room_waypoints_are_inside_their_rooms_and_free():
    from PIL import Image
    g = OccupancyGrid.from_pgm(np.array(Image.open(os.path.join(PKG, 'maps', 'small_house.pgm'))),
                               0.05, -12.5, -12.5)
    for r in ROOMS:
        assert r.contains(*r.waypoint), r.name
        assert g.clearance(*r.waypoint) > 0.4, r.name
    assert g.clearance(0.0, 0.0) > 0.4                        # default spawn point


# ------------------------------------------------------------------ geometry

def test_point_in_polygon_and_approach_point():
    sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert point_in_polygon(1, 1, sq) and not point_in_polygon(3, 1, sq)
    gx, gy, yaw = approach_point(5.0, 0.0, 0.0, 0.0, 0.6)
    assert (gx, gy) == pytest.approx((4.4, 0.0)) and yaw == pytest.approx(0.0)
    gx, gy, _ = approach_point(1.0, 1.0, 1.0, 1.3, 0.6)       # closer than the standoff: stay put-ish
    assert math.hypot(gx - 1.0, gy - 1.3) < 1e-9


def test_grid_clearance_nearest_clear_and_line_of_sight():
    data = np.zeros((40, 40), dtype=int)
    data[:, 20] = 100                                         # a wall at x = 1.0 m
    g = OccupancyGrid(40, 40, 0.05, 0.0, 0.0, data.ravel())
    assert not g.line_of_sight(0.2, 1.0, 1.8, 1.0)
    assert g.line_of_sight(0.2, 1.0, 0.95, 1.0, ignore_last=0.3)
    x, y = g.nearest_clear(1.0, 1.0, 0.3)
    assert g.clearance(x, y) >= 0.3 and abs(x - 1.0) < 0.5


# ------------------------------------------------------------------ projection

def test_lidar_projection_finds_object_not_wall():
    intr = Intrinsics.from_hfov(1.085595, 640, 480)
    # optical frame: z forward, x right. Object 2 m ahead (slightly left), wall 4 m behind it.
    obj = np.array([[x, 0.05, 2.0] for x in np.linspace(-0.3, -0.1, 6)])
    wall = np.array([[x, 0.05, 4.0] for x in np.linspace(-1.0, 1.0, 40)])
    pts = np.vstack([obj, wall])
    u, _ = intr.project(np.array([[-0.2, 0.0, 2.0]]))
    bbox = [u[0] - 30, 100, u[0] + 30, 300]
    p = locate_with_lidar(bbox, pts, intr)
    assert p[2] == pytest.approx(2.0, abs=0.01) and p[0] == pytest.approx(-0.2, abs=0.05)
    assert locate_with_lidar([0, 0, 5, 5], pts, intr) is None    # nothing in those columns


def test_depth_projection():
    intr = Intrinsics(500, 500, 320, 240, 640, 480)
    depth = np.full((480, 480 + 160), 5.0)
    depth[200:280, 400:480] = 1.5
    p = locate_with_depth([400, 200, 480, 280], depth, intr)
    assert p[2] == pytest.approx(1.5) and p[0] == pytest.approx((440 - 320) * 1.5 / 500)


def test_scan_points_and_transforms():
    pts = scan_to_points([1.0, float('inf'), 2.0], 0.0, math.pi / 2, 0.1, 10.0)
    assert pts.shape == (2, 3) and pts[1] == pytest.approx([2.0 * math.cos(math.pi), 2.0 * math.sin(math.pi), 0])
    m = make_transform((1.0, 2.0, 0.0), yaw_to_quaternion(math.pi / 2))
    assert apply_transform(m, [1.0, 0.0, 0.0])[0] == pytest.approx([1.0, 3.0, 0.0])


def test_lidar_projection_thin_chair_legs():
    """A chair seen through its legs: few scattered hits, wall behind. Must pick the legs."""
    intr = Intrinsics.from_hfov(1.085595, 640, 480)
    legs = np.array([[-0.15, 0.1, 1.50], [0.15, 0.1, 1.52], [-0.12, 0.1, 1.85]])
    wall = np.array([[x, 0.1, 3.0] for x in np.linspace(-0.6, 0.6, 25)])
    u, _ = intr.project(np.array([[-0.25, 0, 1.6], [0.25, 0, 1.6]]))
    p = locate_with_lidar([u[0], 100, u[1], 300], np.vstack([legs, wall]), intr)
    assert 1.4 < p[2] < 1.9


def test_strip_robot_from_world():
    sys.path.insert(0, os.path.join(PKG, 'tools'))
    from strip_robot_from_world import strip
    w = ("<world><model name='waffle_pi'><link/></model><model name='Bed_01'/>"
         "<state><model name='waffle_pi'><pose/></model></state></world>")
    out, n = strip(w, 'waffle_pi')
    assert n == 2 and 'waffle_pi' not in out and 'Bed_01' in out


def test_old_map_rooms_aliases_and_objects():
    rooms = load_rooms(yaml.safe_load(open(os.path.join(PKG, 'config', 'rooms_old.yaml'))))
    names = [r.name for r in rooms]
    assert set(names) == {'kitchen', 'dining room', 'bedroom'}
    assert normalize_room('living room', names) == 'dining room'      # alias from rooms_old.yaml
    assert keyword_parse('take me to the lounge', names)[0]['room'] == 'dining room'
    m = SemanticMap(rooms, clock=lambda: 0.0)
    assert load_known_objects(m, yaml.safe_load(open(os.path.join(PKG, 'config', 'known_objects_old.yaml')))) == 9
    assert m.query('fridge')[0]['class'] == 'refrigerator'
    for r in rooms:
        assert r.contains(*r.waypoint)


def test_map_profiles_are_complete():
    prof = yaml.safe_load(open(os.path.join(PKG, 'config', 'map_profiles.yaml')))
    for name, p in prof.items():
        for key in ('map', 'rooms', 'known_objects'):
            f = os.path.join(PKG, 'maps' if key == 'map' else 'config', p[key])
            assert os.path.exists(f), (name, f)


def test_choose_approach_avoids_table_and_walls():
    from semantic_goal_nav.geometry import choose_approach
    data = np.zeros((80, 80), dtype=int)
    data[0, :] = data[-1, :] = data[:, 0] = data[:, -1] = 100       # 4 m x 4 m room
    g = OccupancyGrid(80, 80, 0.05, 0.0, 0.0, data.ravel())
    # chair at (2.0, 3.0), dining table between it and the robot at (2.0, 2.2), robot at (2.0, 0.6)
    gx, gy, yaw = choose_approach(2.0, 3.0, 2.0, 0.6, 0.6, grid=g, avoid=[(2.0, 2.2, 0.8)])
    assert math.hypot(gx - 2.0, gy - 2.2) >= 0.8                     # not under the table
    assert g.clearance(gx, gy) >= 0.3 and 0.55 <= math.hypot(gx - 2.0, gy - 3.0) <= 1.4
    assert abs(math.atan2(3.0 - gy, 2.0 - gx) - yaw) < 1e-6         # facing the chair
    gx, gy, _ = choose_approach(2.0, 2.0, 2.0, 0.6, 0.6, grid=g)     # free room: robot's side
    assert gy < 2.0 and math.hypot(gx - 2.0, gy - 2.0) == pytest.approx(0.6, abs=0.05)
