import numpy as np
from scipy.spatial.transform import Rotation as R

class BBox3D:
    def __init__(self, semantic_id, x_min, y_min, z_min, x_max, y_max, z_max,
                 transform, occlusion_ratio):
        self.semantic_id = semantic_id
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max
        self.transform = np.array(transform).T
        self.occlusion_ratio = occlusion_ratio

    def corners(self):
        return np.array([
            [self.x_min, self.y_min, self.z_min], [self.x_min, self.y_min, self.z_max],
            [self.x_min, self.y_max, self.z_min], [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_min], [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_min], [self.x_max, self.y_max, self.z_max],
        ])

    def world_corners(self):
        ch = np.concatenate([self.corners(), np.ones((8, 1))], axis=1)
        return (self.transform @ ch.T).T[:, :3]

    @property
    def rotation(self):
        return R.from_matrix(self.transform[:3, :3]).as_quat()

    @property
    def translation(self):
        return self.transform[:3, 3]
    
    @property
    def size(self):
        return np.array([self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min])
    
    @property
    def scale_factor(self):
        return np.sqrt(self.transform[0, 0]**2 + self.transform[1, 0]**2)