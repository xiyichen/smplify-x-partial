from collections import defaultdict
import torch
from skimage.transform import estimate_transform
from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
import joblib
from plyfile import PlyData
import numpy as np
import smplx
from smplifyx.utils import _compute_euler_from_matrix
from PIXIE.pixielib.models.SMPLX import SMPLX
from PIXIE.pixielib.utils.config import cfg
from PIXIE.pixielib.utils.util import batch_orth_proj
from PIXIE.pixielib.visualizer import Visualizer
from smplifyx.camera import PerspectiveCamera
import re

class PIXIE(Dataset):
    def __init__(self, root_dir, smplx_path, device):
        matches = []
        for f in glob.glob(os.path.join(root_dir, '**/*_param.pkl'), recursive=True):
            matches.append(f)
        self.matches = matches
        self.device = device
        self.pixie_cfg = cfg
        self.pixie_cfg.model.smplx_model_path = smplx_path
        self.smplx = SMPLX(self.pixie_cfg.model).to(self.device)

    def __len__(self):
        return len(self.matches)

    def transform_points(self, points, tform, points_scale=None):
        points_2d = points[:, :, :2]

        # 'input points must use original range'
        if points_scale:
            assert points_scale[0] == points_scale[1]
            points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]
        # import ipdb; ipdb.set_trace()

        batch_size, n_points, _ = points.shape
        trans_points_2d = torch.bmm(
            torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)],
                      dim=-1),
            tform
        )
        trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
        return trans_points

    def recover_position(self, trans_verts):
        ''' transofrm mesh back to original image space
        '''
        points_scale = (224, 224)
        left = 0
        right = 799
        top = 0
        bottom = 599
        old_size = max(right - left, bottom - top)
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.1)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, 224 - 1], [224 - 1, 0]])
        tform = torch.tensor(estimate_transform('similarity', src_pts, DST_PTS).params, device=self.device).float().unsqueeze(0)
        trans_verts = self.transform_points(trans_verts, tform, points_scale)
        h, w = 600, 800
        trans_verts[:,:,0] = trans_verts[:,:,0]/w*2 - 1
        trans_verts[:,:,1] = trans_verts[:,:,1]/h*2 - 1
        return trans_verts

    def __getitem__(self, idx):
        pkl_fn = self.matches[idx]
        img_name = re.split(r'/|\\', pkl_fn)[-2]
        param_dict = joblib.load(pkl_fn)
        vertices, landmarks, joints = self.smplx(
            shape_params=torch.tensor(param_dict['shape'], device=self.device).unsqueeze(0),
            expression_params=torch.tensor(param_dict['exp'], device=self.device).unsqueeze(0),
            global_pose=torch.tensor(param_dict['global_pose'], device=self.device).unsqueeze(0),
            body_pose=torch.tensor(param_dict['body_pose'], device=self.device).unsqueeze(0),
            jaw_pose=torch.tensor(param_dict['jaw_pose'], device=self.device).unsqueeze(0),
            left_hand_pose=torch.tensor(param_dict['left_hand_pose'], device=self.device).unsqueeze(0),
            right_hand_pose=torch.tensor(param_dict['right_hand_pose'], device=self.device).unsqueeze(0))
        pred_cam = torch.tensor(param_dict['body_cam'], device=self.device)
        left, top, right, bottom = param_dict['bbox']
        focal_length = 1428
        W = 800
        H = 600

        RES = 224

        old_size = max(right - left, bottom - top)

        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.1)

        top_left_x = center[0] - size / 2
        top_left_y = center[1] - size / 2
        bottom_right_x = center[0] + size / 2
        bottom_right_y = center[1] + size / 2

        b = size
        cx = (top_left_x + bottom_right_x) / 2
        cy = (top_left_y + bottom_right_y) / 2

        s = pred_cam[0]

        r = b / RES

        tz = (2 * focal_length) / (r * RES * s)
        cxhat = (2 * (cx - W / 2)) / (s * (b))
        cyhat = (2 * (cy - H / 2)) / (s * (b))
        tx = pred_cam[1] - cxhat
        ty = pred_cam[2] - cyhat
        camera = PerspectiveCamera(translation=torch.tensor([[pred_cam[1], pred_cam[2], 2 * focal_length / (s * b + 1e-9)]]),
                                   center=torch.tensor([[cx, cy]]),
                                   focal_length_x=torch.tensor(focal_length),
                                   focal_length_y=torch.tensor(focal_length))
        projected_vertices = camera(vertices)
        # transformed_vertices = self.recover_position(batch_orth_proj(vertices, cam))
        # transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        # transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        # transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10
        # print(transformed_vertices)


        return vertices, projected_vertices, img_name