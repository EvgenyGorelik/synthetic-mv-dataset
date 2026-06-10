import json, cv2, os, numpy as np

from argparse import ArgumentParser
from synth_ndf_dataset import Dataset


BBOX_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
]


if __name__ == "__main__":
    parser = ArgumentParser(description="NDF dataset viewer tool")
    parser.add_argument("root", type=str, help="root directory of the ndf dataset")
    parser.epilog = "Tools: 'n' for next sample, 'b' for previous sample"
    args = parser.parse_args()

    ds = Dataset(args.root)

    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    idx = 0
    while idx < len(ds):
        sample = ds[idx]
        frame_id = sample.frame_id
        stacked_img = {}
        cam_keys = []
        for cam in ds.cameras:
            img = sample[cam.camera_name].image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            stacked_img[cam] = img
            cam_keys.append(cam)
            for b in sample[cam.camera_name].bbox_3d:
                uv_xy, z = cam.project(b.world_corners())
                for idx1, idx2 in edges:
                    if z[idx1] > 0 and z[idx2] > 0:
                        x1, y1 = uv_xy[idx1]
                        x2, y2 = uv_xy[idx2]
                        cv2.line(stacked_img[cam], (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            for s in sample[cam.camera_name].skeletons:
                for i, j in s.connections():
                    proj_i, z_i = cam.project(s.joint(i).world)
                    proj_j, z_j = cam.project(s.joint(j).world)
                    if z_i[0] > 0 and z_j[0] > 0:
                        x1, y1 = proj_i[0]
                        x2, y2 = proj_j[0]
                        cv2.line(stacked_img[cam], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        stacked_img = [stacked_img[cam] for cam in cam_keys]
        top_row = np.hstack(stacked_img[:2])
        bottom_row = np.hstack(stacked_img[2:4])
        viz_img = np.vstack([top_row, bottom_row])
        viz_img = cv2.putText(viz_img, f'Time: {frame_id} ({idx}/{len(ds)})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        viz_img = cv2.resize(viz_img, (viz_img.shape[1] // 2, viz_img.shape[0] // 2))

        cv2.imshow('Dataset Viewer', viz_img)

        k = None
        while not k:
            k = cv2.waitKey(10)
        if k == ord("n"):
            idx += 1
        elif k == ord("b"):
            idx = max(0, idx - 1)
        elif k == ord(" "):
            idx += 10
        elif k == ord("q"):
            break