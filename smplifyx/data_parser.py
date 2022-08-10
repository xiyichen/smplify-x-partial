# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from glob import glob

import sys
import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_annotation

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(format='coco25', data_folder='data', **kwargs):
    if format.lower() == 'coco25':
        return COCO25(data_folder, **kwargs)
    elif format.lower() == 'halpe':
        return Halpe(data_folder, **kwargs)
    elif format.lower() == 'coco_wholebody':
        return COCO_Wholebody(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(format))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


class COCO25(Dataset):
    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 format='coco25',
                 num_body_joints=25,
                 num_hand_joints=20,
                 **kwargs):
        super(COCO25, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour
        self.num_body_joints = num_body_joints
        self.num_hand_joints = num_hand_joints

        self.format = format

        self.num_joints = (self.num_body_joints +
                           2 * self.num_hand_joints * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)
        self.cnt = 0

    def get_model2data(self):
        return smpl_to_annotation(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                format=self.format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = glob(os.path.join(self.keyp_folder, img_fn + '_*.json'))
        if len(keypoint_fn) == 0:
            raise Exception('Keypoint file for {} does not exist!'.format(img_fn))
        keypoint_fn = keypoint_fn[0]
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 'img': img}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)

class Halpe(COCO25):
    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 format='halpe',
                 **kwargs):

        super(Halpe, self).__init__(data_folder, img_folder=img_folder,
                 keyp_folder=keyp_folder,
                 use_hands=use_hands,
                 use_face=use_face,
                 dtype=dtype,
                 model_type=model_type,
                 joints_to_ign=joints_to_ign,
                 use_face_contour=use_face_contour,
                 format=format,
                 num_body_joints=26,
                 num_hand_joints=20,
                 **kwargs)

    def get_left_shoulder(self):
        return 5

    def get_right_shoulder(self):
        return 6

class COCO_Wholebody(COCO25):
    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 format='coco_wholebody',
                 **kwargs):

        super(COCO_Wholebody, self).__init__(data_folder, img_folder=img_folder,
                 keyp_folder=keyp_folder,
                 use_hands=use_hands,
                 use_face=use_face,
                 dtype=dtype,
                 model_type=model_type,
                 joints_to_ign=joints_to_ign,
                 use_face_contour=use_face_contour,
                 format=format,
                 num_body_joints=23,
                 num_hand_joints=20,
                 **kwargs)

    def get_left_shoulder(self):
        return 5

    def get_right_shoulder(self):
        return 6