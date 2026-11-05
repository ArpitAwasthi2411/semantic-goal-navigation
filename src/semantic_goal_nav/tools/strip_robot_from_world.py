#!/usr/bin/env python3
"""Remove a robot model from a world file saved from Gazebo (File > Save World).

A saved world contains the TurtleBot3 that was spawned into it, so the next launch fails
with "entity [waffle_pi] already exists". This deletes that <model> (and its <state> entry).

    python3 tools/strip_robot_from_world.py my_house.world            # strips waffle_pi
    python3 tools/strip_robot_from_world.py my_house.world burger -o clean.world
"""
import argparse
import re


def strip(text, name):
    n = 0
    for tag in ('model',):
        pattern = re.compile(r"\s*<%s name=['\"]%s['\"]>.*?</%s>" % (tag, re.escape(name), tag), re.S)
        text, k = pattern.subn('', text)
        n += k
    return text, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('world')
    ap.add_argument('robot', nargs='?', default='waffle_pi')
    ap.add_argument('-o', '--output', help='write here instead of in place')
    a = ap.parse_args()
    text, n = strip(open(a.world).read(), a.robot)
    open(a.output or a.world, 'w').write(text)
    print(f"removed {n} '{a.robot}' block(s) -> {a.output or a.world}")


if __name__ == '__main__':
    main()
