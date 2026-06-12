import re
import os

from synth_ndf_dataset.camera_sample import CameraSample
from synth_ndf_dataset.world_sample import WorldSample



class Sample:
    def __init__(self, data_root, frame_id, cameras, palette_person_colors):
        self.data_root = data_root
        self.frame_id = frame_id
        self._cameras = cameras
        self.palette_person_colors = palette_person_colors
        self.world = None
        self.camera = {cam: None for cam in cameras}
        self._load_sample_data()


    def _load_sample_data(self):
        bbox_3d_base = os.path.join(self.data_root, "3d", "bbox_3d", self.frame_id)
        skeleton_frame_id = re.sub(r"^frame_", "", self.frame_id)
        skeleton_path = os.path.join(
            self.data_root, "3d", "skeleton_json", skeleton_frame_id + ".json"
        )
        self.world = WorldSample(
            bbox_3d_path=bbox_3d_base,
            skeleton_json_path=skeleton_path
        )
        for cam in self._cameras:
            base = os.path.join(self.data_root, cam)
            depth_npy = os.path.join(base, "depth", self.frame_id + ".npy")
            self.camera[cam] = CameraSample(
                rgb_path=os.path.join(base, "rgb", self.frame_id + ".png"),
                depth_path=depth_npy if os.path.exists(depth_npy) else None,
                bbox_2d_path=os.path.join(base, "bbox_2d", self.frame_id),
                seg_png_path=os.path.join(base, "segmentation", self.frame_id + ".png"),
                palette_person_colors=self.palette_person_colors,
            )

