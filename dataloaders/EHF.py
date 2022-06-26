from collections import defaultdict
import torch
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

class EHF(Dataset):
    def __init__(self, root_dir, suffix, device, method='smplifyx', **kwargs):
        matches = []
        for f in glob.glob(os.path.join(root_dir, '**/*' + suffix), recursive=True):
            matches.append(f)
        self.root_dir = root_dir
        self.matches = matches
        self.extension = suffix[-4:]
        self.device = device
        self.kwargs = kwargs
        self.method = method

        if suffix == '.ply':
            pkl_fns = []
            for f in glob.glob(os.path.join(self.root_dir, '**/*.pkl'), recursive=True):
                pkl_fns.append(f)
            self.pkl_fns = pkl_fns

        if kwargs.get('use_vposer', True):
            from human_body_prior.tools.model_loader import load_vposer
            vposer_ckpt = os.path.expandvars(kwargs.get('vposer_ckpt'))
            vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
            self.vposer = vposer.to(device=device)
            self.vposer.eval()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        camera_translation = None
        camera_center = None
        img_name = str(idx + 1).zfill(2) + '_cropped'
        focal_length = None
        if self.extension == '.pkl':
            dict = joblib.load(self.matches[idx])
            body_pose = self.vposer.decode(torch.tensor(dict['body_pose'], device=self.device), output_type='aa').view(1, -1)
            global_pose = _compute_euler_from_matrix(torch.tensor(dict['global_orient'], device=self.device))
            jaw_pose = _compute_euler_from_matrix(torch.tensor(dict['jaw_pose'], device=self.device))
            left_hand_pose = dict['left_hand_pose']
            right_hand_pose = dict['right_hand_pose']
            shape_params = dict['betas']
            exp_params = dict['expression']
            new_params = defaultdict(global_orient=global_pose, body_pose=body_pose, jaw_pose=jaw_pose,
                                     left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, betas=shape_params,
                                     expression=exp_params)
            self.kwargs['num_betas'] = shape_params.shape[1]
            self.kwargs['num_expression_coeffs'] = exp_params.shape[1]
            smplx_model = smplx.create(model_path=self.kwargs.get('model_folder'), **self.kwargs)
            smplx_model.reset_params(**new_params)
            body_vertices = smplx_model(return_verts=True).vertices
            camera_translation = dict['camera_translation']
        elif self.extension == '.ply':
            dict = PlyData.read(self.matches[idx])
            body_vertices = dict.elements[0].data
            body_vertices = torch.tensor([list(vertex) for vertex in body_vertices], device=self.device).unsqueeze(0)
            # print(self.pkl_fns)
            if not self.method == 'gt':
                dict_pkl = joblib.load(self.pkl_fns[idx])
                camera_translation = dict_pkl['camera_translation']
        if self.method == 'smplifyx':
            camera_center = [[400, 300]]
            focal_length = 5000
        elif 'pixie' in self.method:
            regression_results_path = self.kwargs.get('pixie_results_directory', None)
            regression_results = joblib.load(os.path.join(regression_results_path, img_name, img_name + '_param.pkl'))
            left, top, right, bottom = regression_results['bbox']

            old_size = max(right - left, bottom - top)

            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * 1.1)

            top_left_x = center[0] - size / 2
            top_left_y = center[1] - size / 2
            bottom_right_x = center[0] + size / 2
            bottom_right_y = center[1] + size / 2
            cx = (top_left_x + bottom_right_x) / 2
            cy = (top_left_y + bottom_right_y) / 2
            camera_center = [[cx, cy]]
            focal_length = 1000


        return body_vertices, camera_translation, camera_center, img_name, focal_length