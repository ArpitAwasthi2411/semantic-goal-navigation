#!/usr/bin/env python3
"""Redraw the house's clean map in the frame of another map (e.g. our earlier apartment_map).

    python3 tools/align_map.py maps/small_house.yaml maps/apartment_map.yaml \
        --world-to-map 5.847 3.653 -97.431 -o maps/apartment_map_aligned

Keeps the target map's frame (so its rooms and marked objects keep their coordinates) but
takes the walls from the clean map, and covers the whole house. --world-to-map is the pose of
the target map frame in the clean map frame (x, y, yaw in degrees).
"""
import argparse
import math
import os

import numpy as np
import yaml
from PIL import Image


def load(path):
    meta = yaml.safe_load(open(path))
    img = np.array(Image.open(os.path.join(os.path.dirname(path), meta['image'])).convert('L'))
    return meta, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('target')
    ap.add_argument('--world-to-map', nargs=3, type=float, required=True)
    ap.add_argument('-o', '--output', required=True, help='output path without extension')
    ap.add_argument('--margin', type=float, default=1.0)
    a = ap.parse_args()
    smeta, simg = load(a.source)
    tmeta, _ = load(a.target)
    res = float(tmeta['resolution'])
    tx, ty, yaw = a.world_to_map[0], a.world_to_map[1], math.radians(a.world_to_map[2])
    c, s = math.cos(yaw), math.sin(yaw)

    # known (non-grey) cells of the source -> bounding box in the target frame
    sres, (sox, soy) = float(smeta['resolution']), smeta['origin'][:2]
    h, w = simg.shape
    rows, cols = np.nonzero(simg != 205)
    wx = sox + (cols + 0.5) * sres
    wy = soy + (h - 1 - rows + 0.5) * sres
    mx = c * (wx - tx) + s * (wy - ty)
    my = -s * (wx - tx) + c * (wy - ty)
    x0, y0 = math.floor((mx.min() - a.margin) / res) * res, math.floor((my.min() - a.margin) / res) * res
    x1, y1 = mx.max() + a.margin, my.max() + a.margin
    W, H = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))

    # every target cell centre -> source pixel (nearest)
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    px = x0 + (jj + 0.5) * res
    py = y0 + (H - 1 - ii + 0.5) * res
    qx = tx + c * px - s * py
    qy = ty + s * px + c * py
    sc = np.floor((qx - sox) / sres).astype(int)
    sr = h - 1 - np.floor((qy - soy) / sres).astype(int)
    ok = (sc >= 0) & (sc < w) & (sr >= 0) & (sr < h)
    out = np.full((H, W), 205, np.uint8)
    out[ok] = simg[sr[ok], sc[ok]]

    Image.fromarray(out).save(a.output + '.pgm')
    meta = {'image': os.path.basename(a.output) + '.pgm', 'mode': 'trinary', 'resolution': res,
            'origin': [round(x0, 3), round(y0, 3), 0.0], 'negate': 0,
            'occupied_thresh': 0.65, 'free_thresh': 0.196}
    with open(a.output + '.yaml', 'w') as f:
        yaml.safe_dump(meta, f, default_flow_style=None, sort_keys=False)
    print(f'{W}x{H} cells, origin ({x0:.2f}, {y0:.2f}) -> {a.output}.yaml')


if __name__ == '__main__':
    main()
