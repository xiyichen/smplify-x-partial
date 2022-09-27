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

from collections import defaultdict
from typing import Dict

import numpy as np

import open3d as o3d
import os
import torch
import torch.nn as nn
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import PIL.Image as pil_img
import trimesh
from human_body_prior.tools.visualization_tools import render_smpl_params, imagearray2file
from human_body_prior.body_model.body_model import BodyModel
import pyrender
from PIL import ImageDraw
from typing import Tuple
from typing import NewType, List, Union
import numpy as np
import torch


__all__ = [
    'Tensor',
    'Array',
]

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)

def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


def smpl_to_annotation(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, format='coco25'):
    ''' Returns the indices of the permutation that maps pose estimation to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        format: bool, optional
            The output format of format estimation. For now only COCO-25, COCO-19, and Halpe for SMPL-X are
            supported. Defaults to 'coco25'

    '''
    if format.lower() == 'halpe':
        if model_type == 'smplx':
            body_mapping = np.array([55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21,
                                     1, 2, 4, 5, 7, 8, 15, 12, 0, 60, 63,
                                     61, 64, 62, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]
            return np.concatenate(mapping)

    elif format.lower() == 'coco_wholebody':
        if model_type == 'smplx':
            body_mapping = np.array([55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21,
                                     1, 2, 4, 5, 7, 8, 60, 61, 62, 63, 64, 65])
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]
            return np.concatenate(mapping)

    elif format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(format))

def get_PIXIE_data(img_path, img_name, detector, device, crop_size=224, hd_size=1024, iscrop=True, scale=1.1):
    image = imread(img_path)[:, :, :3] / 255.
    h, w, _ = image.shape

    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)[None, ...]
    if iscrop:
        bbox = detector.run(image_tensor)
        if bbox is None:
            print('no person detected! run original image')
            left = 0;
            right = w - 1;
            top = 0;
            bottom = h - 1
        else:
            left = bbox[0];
            right = bbox[2];
            top = bbox[1];
            bottom = bbox[3]
        old_size = max(right - left, bottom - top)
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * scale)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
    else:
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        left = 0;
        right = w - 1;
        top = 0;
        bottom = h - 1
        bbox = [left, top, right, bottom]

    # crop image
    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2, 0, 1)
    # hd image
    DST_PTS = np.array([[0,0], [0,hd_size - 1], [hd_size - 1, 0]])
    tform_hd = estimate_transform('similarity', src_pts, DST_PTS)
    hd_image = warp(image, tform_hd.inverse, output_shape=(hd_size, hd_size))
    hd_image = hd_image.transpose(2,0,1)
    # crop image
    return {'image': torch.tensor(dst_image, device=device).float().unsqueeze(0),
            'name': img_name,
            'image_hd': torch.tensor(hd_image, device=device).float().unsqueeze(0),
            'bbox': bbox
            }

def predict_gender_one_img(gender_inferer, img_dir, keypoints_dir):
    import json, cv2
    from homogenus.homogenus.tools.body_cropper import should_accept_pose
    from homogenus.homogenus.tools.image_tools import read_prep_image, cropout_openpose

    Iph = gender_inferer.graph.get_tensor_by_name(u'input_images:0')

    probs_op = gender_inferer.graph.get_tensor_by_name(u'probs_op:0')
    accept_threshold = 0.9
    crop_margin = 0.08

    with open(keypoints_dir, 'r') as f:
        pose_data = json.load(f)

    for opnpose_pIdx in range(1):
        pose_data['people'][opnpose_pIdx]['gender_pd'] = 'neutral'

        pose = np.asarray(pose_data['people'][opnpose_pIdx]['pose_keypoints_2d']).reshape(-1, 3)

        # print(should_accept_pose(pose, human_prob_thr=0.3))
        # exit()
        # if not should_accept_pose(pose, human_prob_thr=0.5):
        #     continue

        crop_info = cropout_openpose(img_dir, pose, want_image=True, crop_margin=crop_margin)
        cropped_image = crop_info['cropped_image']

        if cropped_image.shape[0] < 200 or cropped_image.shape[1] < 200:
            continue

        img = read_prep_image(cropped_image)[np.newaxis]

        probs_ob = gender_inferer.sess.run(probs_op, feed_dict={Iph: img})[0]
        gender_id = np.argmax(probs_ob, axis=0)

        gender_prob = probs_ob[gender_id]
        gender_pd = 'male' if gender_id == 0 else 'female'

        if gender_prob <= accept_threshold:
            gender_pd = 'neutral'

        return gender_pd

def _elementary_basis_vector(axis):
    _AXIS_TO_IND = {'x': 0, 'y': 1, 'z': 2}
    b = torch.zeros(3)
    b[_AXIS_TO_IND[axis]] = 1
    return b

def _compute_euler_from_matrix(dcm, seq='xyz', extrinsic=False):
    # The algorithm assumes intrinsic frame transformations. For representation
    # the paper uses transformation matrices, which are transpose of the
    # direction cosine matrices used by our Rotation class.
    # Adapt the algorithm for our case by
    # 1. Instead of transposing our representation, use the transpose of the
    #    O matrix as defined in the paper, and be careful to swap indices
    # 2. Reversing both axis sequence and angles for extrinsic rotations
    orig_device = dcm.device
    dcm = dcm.to('cpu')
    seq = seq.lower()

    if extrinsic:
        seq = seq[::-1]

    if len(dcm.shape) == 2:
        dcm = dcm[None, :, :]
    num_rotations = dcm.shape[0]

    device = dcm.device

    # Step 0
    # Algorithm assumes axes as column vectors, here we use 1D vectors
    n1 = _elementary_basis_vector(seq[0])
    n2 = _elementary_basis_vector(seq[1])
    n3 = _elementary_basis_vector(seq[2])

    # Step 2
    sl = torch.dot(torch.cross(n1, n2), n3)
    cl = torch.dot(n1, n3)

    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    offset = torch.atan2(sl, cl)
    c = torch.stack((n2, torch.cross(n1, n2), n1)).type(dcm.dtype).to(device)

    # Step 3
    rot = torch.tensor([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ]).type(dcm.dtype)
    # import IPython; IPython.embed(); exit
    res = torch.einsum('ij,...jk->...ik', c, dcm)
    dcm_transformed = torch.einsum('...ij,jk->...ik', res, torch.transpose(c, 0, 1) @ rot)

    # Step 4
    angles = torch.zeros((num_rotations, 3), dtype=dcm.dtype, device=device)

    # Ensure less than unit norm
    positive_unity = dcm_transformed[:, 2, 2] > 1
    negative_unity = dcm_transformed[:, 2, 2] < -1
    dcm_transformed[positive_unity, 2, 2] = 1
    dcm_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = torch.acos(dcm_transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    safe1 = (torch.abs(angles[:, 1]) >= eps)
    safe2 = (torch.abs(angles[:, 1] - np.pi) >= eps)

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = (safe1 > 0) & (safe2 > 0)
    angles[safe_mask, 0] = torch.atan2(dcm_transformed[safe_mask, 0, 2],
                                      -dcm_transformed[safe_mask, 1, 2])
    angles[safe_mask, 2] = torch.atan2(dcm_transformed[safe_mask, 2, 0],
                                      dcm_transformed[safe_mask, 2, 1])
    if extrinsic:
        # For extrinsic, set first angle to zero so that after reversal we
        # ensure that third angle is zero
        # 6a
        angles[~safe_mask, 0] = 0
        # 6b
        angles[~safe1, 2] = torch.atan2(
            dcm_transformed[~safe1, 1, 0] - dcm_transformed[~safe1, 0, 1],
            dcm_transformed[~safe1, 0, 0] + dcm_transformed[~safe1, 1, 1]
        )
        # 6c
        angles[~safe2, 2] = -torch.atan2(
            dcm_transformed[~safe2, 1, 0] + dcm_transformed[~safe2, 0, 1],
            dcm_transformed[~safe2, 0, 0] - dcm_transformed[~safe2, 1, 1]
        )
    else:
        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = torch.atan2(
            dcm_transformed[~safe1, 1, 0] - dcm_transformed[~safe1, 0, 1],
            dcm_transformed[~safe1, 0, 0] + dcm_transformed[~safe1, 1, 1]
        )
        # 6c
        angles[~safe2, 0] = torch.atan2(
            dcm_transformed[~safe2, 1, 0] + dcm_transformed[~safe2, 0, 1],
            dcm_transformed[~safe2, 0, 0] - dcm_transformed[~safe2, 1, 1]
        )

    # Step 7
    if seq[0] == seq[2]:
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = (angles[:, 1] < 0) | (angles[:, 1] > np.pi)
    else:
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = (angles[:, 1] < -np.pi / 2) | (angles[:, 1] > np.pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = (adjust_mask > 0) & (safe_mask > 0)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    # Step 8
    if not torch.all(safe_mask):
        print("Gimbal lock detected. Setting third angle to zero since"
              "it is not possible to uniquely determine all angles.")

    # Reverse role of extrinsic and intrinsic rotations, but let third angle be
    # zero for gimbal locked cases
    if extrinsic:
        # angles = angles[:, ::-1]
        angles = torch.flip(angles, dims=[-1, ])

    angles = angles.to(orig_device)
    return angles

def optimization_visualization(img, smplx_path, vposer, pose_embedding, body_pose, body_model, camera,
                               focal_length, W, H, out_img_save_path, vposer_rendered_img_save_path, use_cuda, mesh_fn, kpts, kpts_gt, **kwargs):
    use_vposer = kwargs.get('use_vposer', True)
    with torch.no_grad():
        # body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
        #
        # model_type = kwargs.get('model_type', 'smpl')
        # append_wrists = model_type == 'smpl' and use_vposer
        # if append_wrists:
        #     wrist_pose = torch.zeros([body_pose.shape[0], 6],
        #                              dtype=body_pose.dtype,
        #                              device=body_pose.device)
        #     body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True, body_pose=body_pose)
        if vposer_rendered_img_save_path:
            bm = BodyModel(smplx_path).to('cuda') if use_cuda else BodyModel(bm_path=smplx_path)
            vposer_rendered_img = render_smpl_params(bm, body_pose.reshape((-1, 21, 3))).reshape(1, 1, 1, 400, 400, 3)
            vposer_rendered_img = imagearray2file(vposer_rendered_img)[0]
            vposer_rendered_img = pil_img.fromarray(vposer_rendered_img)
            vposer_rendered_img.save(vposer_rendered_img_save_path)
        if out_img_save_path:
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()
            out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            out_mesh.apply_transform(rot)
            out_mesh.export(mesh_fn)
            camera_center = camera.center.detach().cpu().numpy().squeeze().copy()
            camera_transl = camera.translation.detach().cpu().numpy().squeeze().copy()
            camera_transl[0] *= -1.0

            output_img = render_mesh(img, out_mesh, camera_center, camera_transl, focal_length, W, H)
            output_img = pil_img.fromarray(output_img)

            draw = ImageDraw.Draw(output_img)
            for x, y in kpts_gt[0]:
                draw.ellipse((x, y, x + 10, y + 10), fill=(255, 0, 0), outline=(0, 0, 0))
            fills = [(255, 192, 103), (0, 0, 255), (0, 255, 0), (255, 255, 255), (255, 255, 0)]
            for idx, (x, y) in enumerate(kpts[0]):
                draw.ellipse((x, y, x + 10, y + 10), fill=fills[idx], outline=(0, 0, 0))

            output_img.save(out_img_save_path)


def render_mesh(img, mesh_trimesh, camera_center, camera_transl, focal_length, img_width, img_height):

    # material = pyrender.MetallicRoughnessMaterial(
    #     metallicFactor=0.0,
    #     alphaMode='OPAQUE',
    #     baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    script_dir = os.path.dirname(os.path.realpath(__file__))
    vertex_colors = np.loadtxt(os.path.join(script_dir, 'smplx_verts_colors.txt'))
    mesh_new = trimesh.Trimesh(vertices=mesh_trimesh.vertices, faces=mesh_trimesh.faces,
                               vertex_colors=vertex_colors)
    mesh_new.vertex_colors = vertex_colors
    print("mesh visual kind: %s" % mesh_new.visual.kind)

    # mesh = pyrender.Mesh.from_points(out_mesh.vertices, colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(mesh_new, smooth=False, wireframe=False)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    # scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_transl

    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)

    light = pyrender.light.DirectionalLight()

    scene.add(light)
    r = pyrender.OffscreenRenderer(viewport_width=img_width,
                                   viewport_height=img_height,
                                   point_size=1.0)
    color, _ = r.render(scene, flags=(pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES))
    # color = color.astype(np.float32) / 255.0
    #
    # output_img = color[:, :, 0:3]
    # output_img = (output_img * 255).astype(np.uint8)

    color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
    color = np.clip(color, 0, 1)

    valid_mask = (color[3] > 0)[np.newaxis]
    output_img = (color[:-1] * valid_mask +
                  (1 - valid_mask) * np.transpose(np.array(img), [2, 0, 1]))
    output_img = np.transpose(output_img, [1, 2, 0])
    output_img = np.clip(output_img, 0, 1)
    output_img = (output_img * 255).astype(np.uint8)

    return output_img

class ProcrustesAlignment(object):
    def __init__(self):
        super(ProcrustesAlignment, self).__init__()

    def __repr__(self):
        return 'ProcrustesAlignment'

    def __call__(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrustes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T
            
        return S1_hat

def mpjpe(input_joints, target_joints):
    ''' Calculate mean per-joint point error
    Parameters
    ----------
        input_joints: numpy.array, Jx3
            The joints predicted by the model
        target_joints: numpy.array, Jx3
            The ground truth joints
    Returns
    -------
        numpy.array, BxJ
            The per joint point error for each element in the batch
    '''

    return np.sqrt(np.power(input_joints - target_joints, 2).sum(axis=-1))

def vertex_to_vertex_error(input_vertices, target_vertices):
    return np.sqrt(np.power(input_vertices - target_vertices, 2).sum(axis=-1))

def np2o3d_pcl(x: np.ndarray) -> o3d.geometry.PointCloud:
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(x)

    return pcl

def point_fscore(
        pred: torch.Tensor,
        gt: torch.Tensor,
        thresh: float) -> Dict[str, float]:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()

    pred_pcl = np2o3d_pcl(pred)
    gt_pcl = np2o3d_pcl(gt)

    gt_to_pred = np.asarray(gt_pcl.compute_point_cloud_distance(pred_pcl))
    pred_to_gt = np.asarray(pred_pcl.compute_point_cloud_distance(gt_pcl))

    recall = (pred_to_gt < thresh).sum() / len(pred_to_gt)
    precision = (gt_to_pred < thresh).sum() / len(gt_to_pred)
    if recall + precision > 0.0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0.0

    return {
        'fscore': fscore,
        'precision': precision,
        'recall': recall,
    }

class PelvisAlignment(object):
    def __init__(self, hips_idxs=None):
        super(PelvisAlignment, self).__init__()
        if hips_idxs is None:
            hips_idxs = [2, 3]
        self.hips_idxs = hips_idxs

    def align_by_pelvis(self, joints):
        pelvis = joints[self.hips_idxs, :].mean(axis=0, keepdims=True)
        return {'joints': joints - pelvis, 'pelvis': pelvis}

    def __call__(self, gt, est):
        gt_out = self.align_by_pelvis(gt)
        est_out = self.align_by_pelvis(est)

        aligned_gt_joints = gt_out['joints']
        aligned_est_joints = est_out['joints']

        return aligned_gt_joints, aligned_est_joints

class PelvisAlignmentMPJPE(PelvisAlignment):
    def __init__(self, fscore_thresholds=None):
        super(PelvisAlignmentMPJPE, self).__init__()
        self.fscore_thresholds = fscore_thresholds

    def __repr__(self):
        msg = [super(PelvisAlignmentMPJPE).__repr__()]
        if self.fscore_thresholds is not None:
            msg.append(
                'F-Score thresholds: ' +
                f'(mm), '.join(map(lambda x: f'{x * 1000}',
                                   self.fscore_thresholds))
            )
        return '\n'.join(msg)

    def __call__(self, est_points, gt_points):
        aligned_gt_points, aligned_est_points = super(
            PelvisAlignmentMPJPE, self).__call__(gt_points, est_points)

        fscore = {}
        if self.fscore_thresholds is not None:
            for thresh in self.fscore_thresholds:
                fscore[thresh] = point_fscore(
                    aligned_est_points, gt_points, thresh)
        return {
            'point': mpjpe(aligned_est_points, aligned_gt_points),
            'fscore': fscore
        }

class ProcrustesAlignmentMPJPE(ProcrustesAlignment):
    def __init__(self, fscore_thresholds=None):
        super(ProcrustesAlignmentMPJPE, self).__init__()
        self.fscore_thresholds = fscore_thresholds

    def __repr__(self):
        msg = [super(ProcrustesAlignment).__repr__()]
        if self.fscore_thresholds is not None:
            msg.append(
                'F-Score thresholds: ' +
                f'(mm), '.join(map(lambda x: f'{x * 1000}',
                                   self.fscore_thresholds))
            )
        return '\n'.join(msg)

    def __call__(self, est_points, gt_points):
        aligned_est_points = super(ProcrustesAlignmentMPJPE, self).__call__(
            est_points, gt_points)

        fscore = {}
        if self.fscore_thresholds is not None:
            for thresh in self.fscore_thresholds:
                fscore[thresh] = point_fscore(
                    aligned_est_points, gt_points, thresh)
        return {
            'point': mpjpe(aligned_est_points, gt_points),
            'fscore': fscore
        }


class ScaleAlignment(object):
    def __init__(self):
        super(ScaleAlignment, self).__init__()

    def __repr__(self):
        return 'ScaleAlignment'

    def __call__(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)
        var2 = np.sum(X2**2)

        # 5. Recover scale.
        scale = np.sqrt(var2 / var1)

        # 6. Recover translation.
        t = mu2 - scale * (mu1)

        # 7. Error:
        S1_hat = scale * S1 + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat

class ProcrustesAlignmentMPJPE(ProcrustesAlignment):
    def __init__(self, fscore_thresholds=None):
        super(ProcrustesAlignmentMPJPE, self).__init__()
        self.fscore_thresholds = fscore_thresholds

    def __repr__(self):
        msg = [super(ProcrustesAlignment).__repr__()]
        if self.fscore_thresholds is not None:
            msg.append(
                'F-Score thresholds: ' +
                f'(mm), '.join(map(lambda x: f'{x * 1000}',
                                   self.fscore_thresholds))
            )
        return '\n'.join(msg)

    def __call__(self, est_points, gt_points):
        aligned_est_points = super(ProcrustesAlignmentMPJPE, self).__call__(
            est_points, gt_points)

        fscore = {}
        if self.fscore_thresholds is not None:
            for thresh in self.fscore_thresholds:
                fscore[thresh] = point_fscore(
                    aligned_est_points, gt_points, thresh)
        return {
            'point': vertex_to_vertex_error(aligned_est_points, gt_points),
            'fscore': fscore
        }


def get_transform(
    center: Array, scale: float,
    res: Tuple[int],
    rot: float = 0
) -> Array:
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3), dtype=np.float32)
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3), dtype=np.float32)
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t.astype(np.float32)