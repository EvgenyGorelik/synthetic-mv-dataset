import cv2
import re


def to_rgba(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


JOINT_CHAIN = [
    "pelvis", "hip", "thigh", "knee", "leg", "calf", "ankle", "foot", "toe",
    "spine", "chest", "neck", "head", "clavicle", "shoulder", "upperarm",
    "arm", "elbow", "forearm", "wrist", "hand",
]


def side(name):
    n = name.lower()
    for tag in ("left", "right", "_l", "_r", "l_", "r_"):
        if tag in n:
            return tag.strip("_")[0]
    return ""


def adjacent_joint(n1, n2):
    if side(n1) != side(n2):
        return False
    a = b = None
    for i, key in enumerate(JOINT_CHAIN):
        if key in n1.lower():
            a = i
        if key in n2.lower():
            b = i
    return a is not None and b is not None and abs(a - b) == 1


def _passenger_key(prim_path):
    m = re.search(r"passenger_\d+", prim_path)
    return m.group(0) if m else prim_path


def _bbox_label_map(raw):
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                sid = int(k)
            except (TypeError, ValueError):
                continue
            out[sid] = v.get("class") if isinstance(v, dict) else v
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            if isinstance(v, dict):
                sid = v.get("semanticId", v.get("semantic_id", i))
                try:
                    sid = int(sid)
                except (TypeError, ValueError):
                    sid = i
                out[sid] = v.get("class")
            else:
                out[i] = v
    return out

