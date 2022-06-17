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

class EHF(Dataset):
    def __init__(self, root_dir, suffix, device, **kwargs):
        matches = []
        for f in glob.glob(os.path.join(root_dir, '**/*' + suffix), recursive=True):
            matches.append(f)
        self.matches = matches
        self.extension = suffix[-4:]
        self.device = device
        self.kwargs = kwargs

        if kwargs.get('use_vposer', True):
            from human_body_prior.tools.model_loader import load_vposer
            vposer_ckpt = os.path.expandvars(kwargs.get('vposer_ckpt'))
            vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
            self.vposer = vposer.to(device=device)
            self.vposer.eval()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        if self.extension == '.pkl':
            dict = joblib.load(self.matches[idx])
            self.smplx_model = smplx.create(model_path=self.kwargs.get('model_folder'), **self.kwargs)
            model_data = np.load(os.path.join(self.kwargs.get('model_folder'), 'smplx', 'SMPLX_MALE.npz'), allow_pickle=True)
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
            self.smplx_model.reset_params(**new_params)
            body_vertices = self.smplx_model(return_verts=True).vertices
        elif self.extension == '.ply':
            dict = PlyData.read(self.matches[idx])
            body_vertices = dict.elements[0].data
            body_vertices = torch.tensor([list(vertex) for vertex in body_vertices], device=self.device).unsqueeze(0)

        return body_vertices