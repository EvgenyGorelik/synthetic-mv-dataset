import numpy as np

class Camera:
    USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, camera_name, resolution, intrinsics,
                 extrinsics=None, extrinsics_opencv=None, **_):
        self.camera_name = camera_name
        self.resolution = tuple(resolution)
        self.K = np.array(intrinsics["K"], dtype=np.float64)
        if extrinsics_opencv is not None:
            self.world_to_camera = np.array(
                extrinsics_opencv["world_to_camera"], dtype=np.float64
            )
        elif extrinsics is not None:
            wtc_row = np.array(extrinsics["world_to_camera"], dtype=np.float64)
            self.world_to_camera = self.USD_TO_OPENCV @ wtc_row.T
        else:
            raise ValueError(f"Camera {camera_name}: cameras.json has no extrinsics.")

    def project(self, world_xyz):
        pts = np.asarray(world_xyz, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        ones = np.ones((pts.shape[0], 1))
        cam = (self.world_to_camera @ np.concatenate([pts, ones], axis=1).T).T
        z = cam[:, 2]
        uv = (self.K @ cam[:, :3].T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uv[:, :2] / uv[:, 2:3]
        return uv, z
