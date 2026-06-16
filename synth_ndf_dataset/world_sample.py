import os
import json
import cv2
import numpy as np
from synth_ndf_dataset.bbox3d import BBox3D
from synth_ndf_dataset.skeleton import Skeleton

class WorldSample:
    def __init__(self, bbox_3d_path,
                 skeleton_json_path):
        self.bbox_3d = self._load_bbox_3d(bbox_3d_path)
        self.skeletons = self._load_skeleton(skeleton_json_path)
        self.paths = {
            "bbox_3d": bbox_3d_path,
            "skeleton": skeleton_json_path,
        }


    def _load_bbox_3d(self, path):
        npy = path + ".npy"
        if not os.path.exists(npy):
            return []
        data = np.load(npy, allow_pickle=True)
        return [
            BBox3D(s, xn, yn, zn, xx, yx, zx, t, o)
            for s, xn, yn, zn, xx, yx, zx, t, o in zip(
                data["semanticId"],
                data["x_min"], data["y_min"], data["z_min"],
                data["x_max"], data["y_max"], data["z_max"],
                data["transform"], data["occlusionRatio"],
            )
        ]

    def _load_skeleton(self, path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            content = json.load(f)
        return [Skeleton(c) for c in content.get("skeletons", [])]
