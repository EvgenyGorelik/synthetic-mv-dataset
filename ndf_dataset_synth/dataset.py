import os
import re
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ndf_dataset_synth.camera import Camera
from ndf_dataset_synth.sample import Sample

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.available_cameras = sorted(
            c for c in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, c))
            and os.path.isdir(os.path.join(dataset_path, c, "rgb"))
        )
        if not self.available_cameras:
            raise RuntimeError(f"No camera folders with rgb/ found under {dataset_path}")
        first = self.available_cameras[0]
        self.samples_ids = sorted(
            f.split(".")[0]
            for f in os.listdir(os.path.join(dataset_path, first, "rgb"))
            if f.endswith(".png")
        )
        with open(os.path.join(dataset_path, "cameras.json")) as f:
            self.metadata = json.load(f)
        self.cameras = [Camera(**c) for c in self.metadata["cameras"]]
        self.palette_person_colors = {}
        palette_file = os.path.join(dataset_path, "segmentation_palette.json")
        if os.path.exists(palette_file):
            with open(palette_file) as f:
                payload = json.load(f)
            for prim_path, rgb in payload.get("person_colors", {}).items():
                self.palette_person_colors[prim_path] = tuple(int(c) for c in rgb)

    def __getitem__(self, idx):
        return Sample(
            self.dataset_path,
            self.samples_ids[idx],
            self.available_cameras,
            self.palette_person_colors,
        )

    def __len__(self):
        return len(self.samples_ids)