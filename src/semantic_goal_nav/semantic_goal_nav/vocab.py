"""Object and room vocabulary shared by every node (pure Python, no ROS).

Canonical class names follow the COCO labels that YOLOv8n outputs, so a detection
can be stored without translation. Everything a person might type ("sofa",
"fridge", "television") is mapped onto one of these names by normalize_class().
"""

# Classes the detector may put on the semantic map. COCO names where one exists.
KNOWN_CLASSES = [
    'chair', 'couch', 'bed', 'dining table', 'tv', 'refrigerator', 'potted plant',
    'bottle', 'cup', 'vase', 'book', 'laptop', 'sports ball', 'clock', 'sink',
    'oven', 'microwave', 'toilet', 'person', 'suitcase', 'backpack',
    # not in COCO; come from the known-objects file or an open-vocabulary/custom model
    'wardrobe', 'cabinet', 'desk', 'door', 'shelf', 'trash can', 'nightstand',
    'coffee table', 'dumbbell',
]

CLASS_SYNONYMS = {
    'chairs': 'chair', 'armchair': 'chair', 'seat': 'chair', 'stool': 'chair',
    'sofa': 'couch', 'sofas': 'couch', 'couches': 'couch', 'settee': 'couch',
    'loveseat': 'couch', 'seating': 'couch',
    'table': 'dining table', 'tables': 'dining table', 'kitchen table': 'dining table',
    'dinner table': 'dining table',
    'tea table': 'coffee table', 'centre table': 'coffee table', 'center table': 'coffee table',
    'television': 'tv', 'tv set': 'tv', 'telly': 'tv', 'tvmonitor': 'tv', 'monitor': 'tv',
    'screen': 'tv',
    'beds': 'bed', 'mattress': 'bed', 'cot': 'bed',
    'fridge': 'refrigerator', 'freezer': 'refrigerator', 'refrigerators': 'refrigerator',
    'plant': 'potted plant', 'plants': 'potted plant', 'flower pot': 'potted plant',
    'pottedplant': 'potted plant',
    'bottles': 'bottle', 'water bottle': 'bottle',
    'ball': 'sports ball', 'football': 'sports ball', 'balls': 'sports ball',
    'closet': 'wardrobe', 'almirah': 'wardrobe', 'cupboard': 'wardrobe',
    'kitchen cabinet': 'cabinet', 'cabinets': 'cabinet', 'drawer': 'cabinet', 'drawers': 'cabinet',
    'tv cabinet': 'cabinet',
    'study table': 'desk', 'reading desk': 'desk', 'work desk': 'desk', 'desks': 'desk',
    'bookshelf': 'shelf', 'bookcase': 'shelf', 'rack': 'shelf', 'shoe rack': 'shelf',
    'shelves': 'shelf',
    'dustbin': 'trash can', 'bin': 'trash can', 'trash': 'trash can', 'garbage can': 'trash can',
    'trashcan': 'trash can',
    'night stand': 'nightstand', 'bedside table': 'nightstand',
    'doors': 'door', 'doorway': 'door',
    'people': 'person', 'human': 'person', 'man': 'person', 'woman': 'person',
    'weights': 'dumbbell', 'dumbbells': 'dumbbell',
    'stove': 'oven', 'cooker': 'oven',
}

# Rooms of the AWS small house (polygons live in config/rooms.yaml).
ROOM_SYNONYMS = {
    'kitchen area': 'kitchen', 'dining room': 'kitchen', 'dining area': 'kitchen',
    'living': 'living room', 'livingroom': 'living room', 'lounge': 'living room',
    'hall': 'living room', 'drawing room': 'living room', 'sitting room': 'living room',
    'bed room': 'bedroom', 'master bedroom': 'bedroom',
    'gym': 'gym', 'balcony': 'gym', 'exercise room': 'gym', 'fitness room': 'gym',
    'study': 'study', 'study room': 'study', 'tv room': 'study', 'office': 'study',
}


def _clean(text):
    return ' '.join(str(text).strip().lower().replace('_', ' ').replace('-', ' ').split())


def normalize_class(name):
    """Map any spelling onto a KNOWN_CLASSES name, or None if it is not an object we know."""
    if not name:
        return None
    c = _clean(name)
    if c in KNOWN_CLASSES:
        return c
    if c in CLASS_SYNONYMS:
        return CLASS_SYNONYMS[c]
    for strip in ('the ', 'a ', 'an '):
        if c.startswith(strip):
            return normalize_class(c[len(strip):])
    if c.endswith('es') and c[:-2] in KNOWN_CLASSES:
        return c[:-2]
    if c.endswith('s') and c[:-1] in KNOWN_CLASSES:
        return c[:-1]
    return None


# Extra names for the rooms of the map in use (rooms.yaml "aliases"); they win over ROOM_SYNONYMS.
ROOM_ALIASES = {}


def register_room_aliases(aliases):
    """aliases: {alias: room name}. Called when a rooms file / semantic map is loaded."""
    for k, v in (aliases or {}).items():
        ROOM_ALIASES[_clean(k)] = v


def normalize_room(name, rooms):
    """Map a spoken room name onto one of `rooms` (the names in rooms.yaml), or None."""
    if not name:
        return None
    c = _clean(name)
    if c.startswith('the '):
        c = c[4:]
    names = {r.lower(): r for r in rooms}
    if c in names:
        return names[c]
    if ROOM_ALIASES.get(c, '').lower() in names:
        return names[ROOM_ALIASES[c].lower()]
    s = ROOM_SYNONYMS.get(c)
    if s and s in names:
        return names[s]
    return None


def all_object_phrases():
    """(phrase, canonical) pairs, longest phrase first, for keyword matching in free text."""
    pairs = [(c, c) for c in KNOWN_CLASSES] + list(CLASS_SYNONYMS.items())
    pairs += [(c + 's', c) for c in KNOWN_CLASSES if not c.endswith('s')]
    return sorted(set(pairs), key=lambda p: -len(p[0]))


def all_room_phrases(rooms):
    pairs = [(r.lower(), r) for r in rooms]
    names = {r.lower(): r for r in rooms}
    taken = {k for k, _ in pairs}
    for k, v in ROOM_ALIASES.items():
        if v.lower() in names and k not in taken:
            pairs.append((k, names[v.lower()]))
            taken.add(k)
    pairs += [(k, names[v]) for k, v in ROOM_SYNONYMS.items() if v in names and k not in taken]
    return sorted(set(pairs), key=lambda p: -len(p[0]))
