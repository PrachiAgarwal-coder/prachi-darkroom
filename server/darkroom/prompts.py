"""Prompt -> objects, attributes and spatial relations.

Used by the semantic verifier: each object becomes a detector query
("a red cube"), and each relation becomes a geometric check on the boxes.
Rule-based on purpose: it is fast, deterministic and explainable in the viva.
"""
import re

COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black", "brown",
          "gray", "grey", "gold", "golden", "silver", "teal", "maroon"}
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4}
# longest phrases first so "to the left of" wins over "left of"
RELATIONS = [("to the left of", "left"), ("left of", "left"), ("to the right of", "right"), ("right of", "right"),
             ("on top of", "on"), ("above", "above"), ("over", "above"), ("under", "below"), ("below", "below"),
             ("beneath", "below"), ("in front of", "front"), ("behind", "behind"), ("next to", "beside"),
             ("beside", "beside"), ("on", "on")]
RELATION_WORDS = {"left": "left of", "right": "right of", "on": "on top of", "above": "above", "below": "below",
                  "front": "in front of", "behind": "behind", "beside": "beside"}
STOP = {"a", "an", "the", "of", "and", "with", "in", "at", "is", "are", "to", "its", "their", "some",
        "photo", "picture", "image", "style", "lighting", "realistic", "detailed", "background", "scene"}
ADJ = {"small", "big", "large", "tiny", "wooden", "metal", "glass", "shiny", "old", "new", "tall", "short",
       "round", "square", "fluffy", "cute", "modern", "ancient"}
SCENE_WORDS = {"sunset", "night", "beach", "ocean", "forest", "desert", "snow", "city", "space", "studio",
               "table", "kitchen", "room", "street", "sky", "field", "garden", "park", "road", "day", "morning"}


def parse(prompt: str):
    text = re.sub(r"[^a-z\s]", " ", prompt.lower())
    words = text.split()
    objects, relations = [], []
    attrs, count, pending = [], 1, None
    i = 0
    while i < len(words):
        rel = None
        for phrase, r in RELATIONS:
            p = phrase.split()
            if words[i:i + len(p)] == p:
                rel = (len(p), r)
                break
        if rel and objects:
            pending = (len(objects) - 1, rel[1])
            attrs = []
            i += rel[0]
            continue
        w = words[i]
        if w in NUMBERS:
            count = NUMBERS[w]
        elif w in COLORS or w in ADJ:
            attrs.append(w)
        elif w in STOP or w in SCENE_WORDS:
            if w in SCENE_WORDS:
                attrs = []
        elif len(w) > 2:
            # treat any other content word as a noun; merge compounds like "auto rickshaw"
            if i + 1 < len(words) and words[i + 1] not in STOP and words[i + 1] not in COLORS \
                    and not any(words[i + 1:i + 1 + len(p.split())] == p.split() for p, _ in RELATIONS) \
                    and words[i + 1] not in SCENE_WORDS and len(words[i + 1]) > 2:
                w = w + " " + words[i + 1]
                i += 1
            noun = w[:-1] if count > 1 and w.endswith("s") else w
            for _ in range(count):
                objects.append({"noun": noun, "attrs": list(attrs)})
            if pending:
                relations.append({"a": pending[0], "rel": pending[1], "b": len(objects) - 1})
                pending = None
            attrs, count = [], 1
        i += 1
    return {"objects": objects, "relations": relations}


def phrase(obj):
    return " ".join(obj["attrs"] + [obj["noun"]])


def check_relation(rel, box_a, box_b):
    """Boxes are [x0, y0, x1, y1] normalised to 0..1. Returns True if the relation holds."""
    ax, ay = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    bx, by = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    if rel == "left":
        return ax < bx
    if rel == "right":
        return ax > bx
    if rel in ("on", "above"):
        return ay < by
    if rel == "below":
        return ay > by
    if rel == "front":
        return box_a[3] >= box_b[3] - 0.02     # lower bottom edge = closer to camera
    if rel == "behind":
        return box_a[3] < box_b[3] + 0.02
    if rel == "beside":
        return abs(ay - by) < 0.25
    return True
