"""Natural language -> structured navigation steps (pure Python, no ROS).

Two modes, as in the report:
  keyword  string matching + synonym table ("sofa" -> couch). Offline, always works.
  llm      a local Ollama model gets a structured prompt and returns JSON. The JSON is
           validated against the known classes and rooms, so the LLM can never send
           the robot to something that does not exist. If the LLM call fails or
           returns nonsense, the node falls back to keyword mode.

A step is a dict:
  {'action': 'goto' | 'goto_room' | 'list' | 'count' | 'describe' | 'stop',
   'target': class or None, 'room': room or None, 'near': class or None,
   'exclude': class or None, 'modifier': 'nearest'|'farthest'|'first'|... or None}
"""
import json
import re

from .vocab import KNOWN_CLASSES, all_object_phrases, all_room_phrases, normalize_class, normalize_room

ACTIONS = ('goto', 'goto_room', 'list', 'count', 'describe', 'stop')
MODIFIERS = ('nearest', 'farthest', 'first', 'second', 'third', 'fourth', 'fifth')

_STOP = re.compile(r'\b(stop|halt|cancel|abort|freeze|stay)\b')
_DESCRIBE = re.compile(r'\b(what (do|can) you see|what have you seen|what is (on|in) the map|'
                       r'what objects|show (me )?(the )?map|describe)\b')
_COUNT = re.compile(r'\b(how many|count|number of)\b')
_LIST = re.compile(r'\b(list|where (is|are)|find|locate|show)\b')
_NEAR = re.compile(r'\b(near|next to|beside|by|close to|nearest to|around)\s+(the\s+|a\s+)?')
_EXCLUDE = re.compile(r'\b(avoid|away from|not near|except|but not)\s+(the\s+|a\s+)?')
_MOD = {
    'nearest': r'\b(nearest|closest)\b', 'farthest': r'\b(farthest|furthest)\b',
    'first': r'\b(first|1st)\b', 'second': r'\b(second|2nd)\b', 'third': r'\b(third|3rd)\b',
    'fourth': r'\b(fourth|4th)\b', 'fifth': r'\b(fifth|5th)\b',
}


def empty_step(action='goto'):
    return {'action': action, 'target': None, 'room': None, 'near': None,
            'exclude': None, 'modifier': None}


def _find_phrases(text, pairs):
    """All (start, end, canonical) matches, longest phrases win, no overlaps."""
    taken = [False] * len(text)
    found = []
    for phrase, canon in pairs:
        for m in re.finditer(r'\b' + re.escape(phrase) + r'\b', text):
            s, e = m.span()
            if any(taken[s:e]):
                continue
            for i in range(s, e):
                taken[i] = True
            found.append((s, e, canon))
    return sorted(found)


def _split_sentences(text):
    parts = re.split(r'\s*(?:,\s*)?\b(?:and then|then|after that|afterwards)\b\s*|\s*;\s*|\.\s+', text)
    return [p.strip(' .,!?') for p in parts if p and p.strip(' .,!?')]


def keyword_parse(text, rooms):
    """Free text -> list of steps, using string matching only. [] if nothing understood."""
    steps = []
    for part in _split_sentences(text.lower()):
        s = _keyword_one(part, rooms)
        if s:
            steps.append(s)
    return steps


def _keyword_one(text, rooms):
    if _STOP.search(text) and not re.search(r'\b(go|take|navigate|move)\b', text):
        return empty_step('stop')
    if _DESCRIBE.search(text):
        return empty_step('describe')

    room_hits = _find_phrases(text, all_room_phrases(rooms))
    objs = [o for o in _find_phrases(text, all_object_phrases())       # "tv room" is a room, not a tv
            if not any(o[0] < re_ and rs < o[1] for rs, re_, _ in room_hits)]
    step = empty_step()

    # qualifiers: "near the X", "avoid the X" refer to the object right after the keyword
    near_spans = [m.end() for m in _NEAR.finditer(text)]
    excl_spans = [m.end() for m in _EXCLUDE.finditer(text)]
    main = []
    for s, e, c in objs:
        if any(s == p for p in near_spans):
            step['near'] = c
        elif any(s == p for p in excl_spans):
            step['exclude'] = c
        else:
            main.append(c)
    if room_hits:
        step['room'] = room_hits[0][2]
    for mod, pat in _MOD.items():
        if re.search(pat, text):
            step['modifier'] = mod
            break

    if _COUNT.search(text):
        step['action'] = 'count'
    elif re.search(r'\b(list|where (is|are)|which)\b', text) and main:
        step['action'] = 'list'

    if main:
        step['target'] = main[0]
    elif step['action'] in ('count', 'list'):
        return None
    elif step['room']:
        step['action'] = 'goto_room'
    else:
        return None
    if step['action'] == 'goto' and step['modifier'] is None:
        step['modifier'] = 'nearest'
    return step


# ------------------------------------------------------------------ LLM mode

def build_prompt(text, rooms, seen_classes=None):
    classes = ', '.join(KNOWN_CLASSES)
    room_list = ', '.join(rooms) or '(none)'
    seen = ', '.join(seen_classes) if seen_classes else '(nothing yet)'
    return f"""You turn a person's instruction to a home robot into JSON. Output ONLY one JSON object, no other text.

Object classes (use exactly these strings): {classes}
Rooms (use exactly these strings): {room_list}
Objects currently on the robot's map: {seen}

Schema:
{{"steps": [{{"action": "goto" | "goto_room" | "list" | "count" | "describe" | "stop",
  "target": "<object class or null>", "room": "<room or null>",
  "near": "<object class the target must be near, or null>",
  "exclude": "<object class to stay away from, or null>",
  "modifier": "nearest" | "farthest" | "first" | "second" | "third" | null}}]}}

Rules:
- goto: go to an object. target is required. Put a room in "room" if the person names one.
- goto_room: go to a room with no object named. target must be null.
- list / count: questions like "where are the chairs" or "how many chairs".
- describe: "what do you see". stop: "stop", "cancel".
- Several instructions in one sentence -> several steps, in order.
- Words like sofa, fridge, television, table must be mapped to the class list (couch, refrigerator, tv, dining table).
- Never invent a class or a room that is not in the lists.

Examples:
"go to the chair" -> {{"steps": [{{"action": "goto", "target": "chair", "room": null, "near": null, "exclude": null, "modifier": "nearest"}}]}}
"take me to the sofa in the living room" -> {{"steps": [{{"action": "goto", "target": "couch", "room": "living room", "near": null, "exclude": null, "modifier": "nearest"}}]}}
"go to the chair near the table" -> {{"steps": [{{"action": "goto", "target": "chair", "room": null, "near": "dining table", "exclude": null, "modifier": "nearest"}}]}}
"go to the kitchen" -> {{"steps": [{{"action": "goto_room", "target": null, "room": "kitchen", "near": null, "exclude": null, "modifier": null}}]}}
"how many chairs are in the kitchen" -> {{"steps": [{{"action": "count", "target": "chair", "room": "kitchen", "near": null, "exclude": null, "modifier": null}}]}}
"go to the bed and then come to the fridge" -> {{"steps": [{{"action": "goto", "target": "bed", "room": null, "near": null, "exclude": null, "modifier": "nearest"}}, {{"action": "goto", "target": "refrigerator", "room": null, "near": null, "exclude": null, "modifier": "nearest"}}]}}

Instruction: "{text}"
JSON:"""


def extract_json(text):
    """First balanced {...} block in a model reply (handles ``` fences and chatter)."""
    if not text:
        return None
    text = text.replace('```json', '').replace('```', '')
    start = text.find('{')
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find('{', start + 1)
    return None


def validate_llm_steps(data, rooms):
    """Check and normalise LLM JSON. Returns (steps, problems). Invalid steps are dropped."""
    problems = []
    if isinstance(data, dict) and 'steps' not in data and 'action' in data:
        data = {'steps': [data]}
    if not isinstance(data, dict) or not isinstance(data.get('steps'), list):
        return [], ['reply has no "steps" list']
    steps = []
    for raw in data['steps']:
        if not isinstance(raw, dict):
            problems.append(f'step is not an object: {raw!r}')
            continue
        s = empty_step(str(raw.get('action') or '').strip().lower())
        if s['action'] not in ACTIONS:
            problems.append(f'unknown action {raw.get("action")!r}')
            continue
        tgt = raw.get('target', raw.get('target_class'))
        for key, val in (('target', tgt), ('near', raw.get('near', raw.get('reference_class'))),
                         ('exclude', raw.get('exclude', raw.get('exclude_class')))):
            if val in (None, '', 'null', 'none'):
                continue
            c = normalize_class(val)
            if c is None:
                problems.append(f'unknown object {val!r}')
            s[key] = c
        room = raw.get('room')
        if room not in (None, '', 'null', 'none'):
            s['room'] = normalize_room(room, rooms)
            if s['room'] is None:
                problems.append(f'unknown room {room!r}')
        mod = raw.get('modifier')
        if isinstance(mod, str) and mod.lower() in MODIFIERS:
            s['modifier'] = mod.lower()
        if s['action'] == 'goto' and not s['target']:
            if s['room']:
                s['action'] = 'goto_room'
            else:
                problems.append('goto without a valid target')
                continue
        if s['action'] == 'goto_room' and not s['room']:
            problems.append('goto_room without a valid room')
            continue
        if s['action'] in ('list', 'count') and not s['target']:
            problems.append(f'{s["action"]} without a valid target')
            continue
        if s['action'] == 'goto' and s['modifier'] is None:
            s['modifier'] = 'nearest'
        steps.append(s)
    return steps, problems


def describe_step(s):
    a = s['action']
    if a == 'goto_room':
        return f'go to the {s["room"]}'
    if a in ('stop', 'describe'):
        return a
    t = s['target']
    out = {'goto': f'go to the {s.get("modifier") or "nearest"} {t}',
           'list': f'list every {t}', 'count': f'count the {t}s'}[a]
    if s.get('near'):
        out += f' near the {s["near"]}'
    if s.get('room'):
        out += f' in the {s["room"]}'
    if s.get('exclude'):
        out += f' away from the {s["exclude"]}'
    return out
