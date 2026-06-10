import os
import json
import cv2
import numpy as np
from ndf_dataset_synth.utils import _passenger_key, _bbox_label_map
from ndf_dataset_synth.bbox2d import BBox2D
from ndf_dataset_synth.bbox3d import BBox3D

class CameraSample:
    def __init__(self, rgb_path, depth_path, bbox_2d_path, bbox_3d_path,
                 skeleton_json_path, seg_png_path, palette_person_colors):
        self.image = cv2.imread(rgb_path, cv2.IMREAD_COLOR_RGB)
        assert self.image is not None, f"Failed to load image at {rgb_path}"
        self.depth = self._load_depth(depth_path)
        self.bbox_2d = self._load_bbox_2d(bbox_2d_path)
        self.bbox_3d = self._load_bbox_3d(bbox_3d_path)
        self.skeleton = self._load_skeleton(skeleton_json_path)
        self.person_masks, self.seg_mapping = self._load_segmentation(
            seg_png_path, palette_person_colors
        )

    def _load_depth(self, path):
        if path is None or not os.path.exists(path):
            return None
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def _load_bbox_2d(self, path):
        npy = path + ".npy"
        if not os.path.exists(npy):
            return []
        data = np.load(npy, allow_pickle=True)
        labels = {}
        labels_json = path + ".json"
        if os.path.exists(labels_json):
            with open(labels_json) as f:
                labels = _bbox_label_map(json.load(f))
        return [
            BBox2D(int(xn), int(yn), int(xx), int(yx), int(sid), labels.get(int(sid)))
            for xn, yn, xx, yx, sid in zip(
                data["x_min"], data["y_min"], data["x_max"], data["y_max"],
                data["semanticId"],
            )
        ]

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
            return None
        with open(path) as f:
            return json.load(f)

    def _load_segmentation(self, png_path, palette_person_colors):
        if not os.path.exists(png_path) or not palette_person_colors:
            return {}, {}
        seg = cv2.imread(png_path, cv2.IMREAD_COLOR_RGB)
        if seg is None:
            return {}, {}
        masks = {}
        mapping = {}
        for prim_path, rgb in palette_person_colors.items():
            color = np.array(rgb, dtype=seg.dtype)
            px = np.all(seg == color, axis=-1)
            if not px.any():
                continue
            mapping[tuple(int(c) for c in rgb)] = prim_path
            actor = _passenger_key(prim_path)
            masks[actor] = masks.get(actor, np.zeros(seg.shape[:2], bool)) | px
        return masks, mapping

