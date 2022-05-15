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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2

from optimizers import optim_factory

import fitting
import PIL.Image as pil_img
from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.visualization_tools import render_smpl_params, imagearray2file
from human_body_prior.body_model.body_model import BodyModel
from utils import _compute_euler_from_matrix, optimization_visualization
from sys import platform
if platform == 'linux' or platform == "linux2":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def fit_single_frame(img,
                     keypoints,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     output_folder='.',
                     img_name='',
                     regression_results=None,
                     smplx_path='./smplx_model/models/smplx/SMPLX_MALE.npz',
                     curr_img_folder='.',
                     **kwargs):

    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    img = torch.tensor(img, dtype=dtype)

    H, W, _ = img.shape

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    if regression_results:
        model_data = np.load(smplx_path, allow_pickle=True)
        regression_pose = regression_results['body_pose']
        regression_pose = [_compute_euler_from_matrix(torch.tensor(joint_rotation, device=device)) for joint_rotation in regression_pose]
        regression_pose = torch.cat(regression_pose).reshape(1, -1)

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()
        if regression_results:
            pose_embedding_init = torch.tensor(vposer.encode(regression_pose).sample(), dtype=dtype, device=device,
                                               requires_grad=True)
            pose_embedding = torch.tensor(pose_embedding_init, dtype=dtype, device=device,
                                          requires_grad=True)

            if visualize:
                with torch.no_grad():
                    vposer_rendered_img_save_path = os.path.join(curr_img_folder, '{}_regression_pose.png'.format(img_name))
                    bm = BodyModel(smplx_path).to('cuda') if use_cuda else BodyModel(bm_path=smplx_path)
                    vposer_rendered_img = render_smpl_params(bm, regression_pose.reshape((-1, 21, 3))).reshape(1, 1, 1, 400, 400, 3)
                    vposer_rendered_img = imagearray2file(vposer_rendered_img)[0]
                    vposer_rendered_img = pil_img.fromarray(vposer_rendered_img)
                    vposer_rendered_img.save(vposer_rendered_img_save_path)
                    print("saved regression pose to %s" % vposer_rendered_img_save_path)
        else:
            pose_embedding_init = torch.zeros([batch_size, vposer_latent_dim],
                                              dtype=dtype, device=device,
                                              requires_grad=True)
            pose_embedding = pose_embedding_init

    if regression_results:
        global_pose = _compute_euler_from_matrix(torch.tensor(regression_results['global_pose'], device=device))

        jaw_pose = _compute_euler_from_matrix(torch.tensor(regression_results['jaw_pose'], device=device))

        left_hand_pose = regression_results['left_hand_pose']
        left_hand_pose = [_compute_euler_from_matrix(torch.tensor(joint_rotation, device=device)) for joint_rotation in
                          left_hand_pose]
        left_hand_pose = torch.cat(left_hand_pose).reshape(-1, 1)
        num_pca_comps = kwargs.get('num_pca_comps')
        left_hand_components = torch.tensor(model_data['hands_componentsl'][:num_pca_comps], device=device)
        left_hand_pose = (left_hand_components @ left_hand_pose).reshape(1, -1)

        right_hand_pose = regression_results['right_hand_pose']
        right_hand_pose = [_compute_euler_from_matrix(torch.tensor(joint_rotation, device=device)) for joint_rotation in
                           right_hand_pose]
        right_hand_pose = torch.cat(right_hand_pose).reshape(-1, 1)
        right_hand_components = torch.tensor(model_data['hands_componentsr'][:num_pca_comps], device=device)
        right_hand_pose = (right_hand_components @ right_hand_pose).reshape(1, -1)

        shape_params = regression_results['shape']
        exp_params = regression_results['exp']

        new_params = defaultdict(global_orient=global_pose, body_pose=pose_embedding, jaw_pose=jaw_pose,
                                 left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, betas=shape_params,
                                 expression=exp_params)
    else:
        new_params = defaultdict(body_pose=pose_embedding)
    body_model.reset_params(**new_params)

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    indices_low_conf = [i for i in range(len(joints_conf[0])) if joints_conf[0][i] < 0.2]
    joint_weights[:, indices_low_conf] = 0
    indices_5kpts = [2, 5, 8, 15, 16]

    init_joints_idxs_trimmed = []
    for idx in init_joints_idxs:
        if gt_joints[0, idx, 0] != 0 and gt_joints[0, idx, 1] != 0:
            init_joints_idxs_trimmed.append(idx)

    init_joints_idxs = init_joints_idxs_trimmed

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')
    if kwargs.get('use_camera_prior') and regression_results:
        left, top, right, bottom = regression_results['bbox']

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

        pred_cam = regression_results['body_cam']
        s = pred_cam[0]

        r = b / RES

        tz = (2 * focal_length) / (r * RES * s)
        cxhat = (2 * (cx - W / 2)) / (s * (b))
        cyhat = (2 * (cy - H / 2)) / (s * (b))
        tx = pred_cam[1] - cxhat
        ty = pred_cam[2] - cyhat
        init_t = torch.tensor([[pred_cam[1], pred_cam[2], 2 * focal_length / (s * b + 1e-9)]], dtype=dtype, device=device)
        with torch.no_grad():
            camera.translation[:] = torch.tensor(init_t.reshape(1, -1), device=device)
            camera.center[:] = torch.tensor([cx, cy], dtype=dtype, device=device)

    else:
        init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length, dtype=dtype)
        with torch.no_grad():
            camera.translation[:] = torch.tensor(init_t.reshape(1, -1), device=device)
            camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

    if visualize:
        with torch.no_grad():
            out_img_save_path = os.path.join(curr_img_folder, '{}_init_cam.png'.format(img_name))
            vposer_rendered_img_save_path = None
            optimization_visualization(img, smplx_path, vposer, pose_embedding, body_model, camera,
                                       focal_length, W, H, out_img_save_path, vposer_rendered_img_save_path,
                                       use_cuda, mesh_fn, **kwargs)

            print("saved rendered mesh and vposer for step 1 to %s" % out_img_save_path)

            body_pose = vposer.decode(pose_embedding_init, output_type='aa').view(1, -1) if use_vposer else None
            body_model_output = body_model(return_verts=True,
                                           body_pose=body_pose)
            projected_joints = camera(body_model_output.joints)
            projected_joints_5kpts_arr = projected_joints[:, indices_5kpts, :]
            gt_joints_5kpts_arr = gt_joints[:, indices_5kpts]
            from PIL import Image, ImageDraw
            im = Image.open(osp.join(kwargs.get('data_folder'), 'images', img_name + '.jpg'))
            draw = ImageDraw.Draw(im)
            for x, y in gt_joints_5kpts_arr[0]:
                draw.ellipse((x, y, x + 10, y + 10), fill=(255, 0, 0), outline=(0, 0, 0))
            fills = [(255, 192, 103), (0, 0, 255), (0, 255, 0), (255, 255, 255), (255, 255, 0)]
            for idx, (x, y) in enumerate(projected_joints_5kpts_arr[0]):
                draw.ellipse((x, y, x + 10, y + 10), fill=fills[idx], outline=(0, 0, 0))
            im.save(os.path.join(curr_img_folder, '{}_5kpts_positions_init_cam.png'.format(img_name)))

    camera_loss = fitting.create_loss('camera_init',
                                      joints_conf=joints_conf,
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=False, **kwargs) as monitor:

        data_weight = 1000 / H
        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        # Reset the parameters to estimate the initial translation of the
        # body model
        # body_model.reset_params(body_pose=body_mean_pose)

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # Update the value of the translation of the camera as well as
        # the image center.

        # Re-enable gradient calculation for the camera translation
        camera.translation.requires_grad = True
        camera.rotation.requires_grad = True
        body_model.global_orient.requires_grad = True

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False, return_verts=False)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        if visualize:
            with torch.no_grad():
                out_img_save_path = os.path.join(curr_img_folder, '{}_step1.png'.format(img_name))
                vposer_rendered_img_save_path = os.path.join(curr_img_folder, '{}_vposer_step1.png'.format(img_name))
                optimization_visualization(img, smplx_path, vposer, pose_embedding, body_model, camera,
                                           focal_length, W, H, out_img_save_path, vposer_rendered_img_save_path,
                                           use_cuda, mesh_fn, **kwargs)

                print("saved rendered mesh and vposer for step 1 to %s" % out_img_save_path)

                body_pose = vposer.decode(pose_embedding_init, output_type='aa').view(1, -1) if use_vposer else None
                body_model_output = body_model(return_verts=True,
                                               body_pose=body_pose)
                projected_joints = camera(body_model_output.joints)
                projected_joints_5kpts_arr = projected_joints[:, indices_5kpts, :]
                gt_joints_5kpts_arr = gt_joints[:, indices_5kpts]
                from PIL import Image, ImageDraw
                im = Image.open(osp.join(kwargs.get('data_folder'), 'images', img_name + '.jpg'))
                draw = ImageDraw.Draw(im)
                for x, y in gt_joints_5kpts_arr[0]:
                    draw.ellipse((x, y, x + 10, y + 10), fill=(255, 0, 0), outline=(0, 0, 0))
                fills = [(255, 192, 103), (0, 0, 255), (0, 255, 0), (255, 255, 255), (255, 255, 0)]
                for idx, (x, y) in enumerate(projected_joints_5kpts_arr[0]):
                    draw.ellipse((x, y, x + 10, y + 10), fill=fills[idx], outline=(0, 0, 0))
                im.save(os.path.join(curr_img_folder, '{}_5kpts_positions_after_cam_opt.png'.format(img_name)))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient,
                                     body_pose=pose_embedding_init)
            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    for i in range(pose_embedding.shape[1]):
                        pose_embedding[:, i] = pose_embedding_init[:, i]

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

                if visualize:
                    with torch.no_grad():
                        out_img_save_path = os.path.join(curr_img_folder, '{}_{:02d}.png'.format(img_name, opt_idx))
                        vposer_rendered_img_save_path = os.path.join(curr_img_folder,
                                                                     '{}_vposer_{:02d}.png'.format(img_name, opt_idx))
                        optimization_visualization(img, smplx_path, vposer, pose_embedding, body_model, camera,
                                                   focal_length, W, H, out_img_save_path, vposer_rendered_img_save_path,
                                                   use_cuda, mesh_fn, **kwargs)
                        print("saved rendered mesh and vposer for current stage to %s" % out_img_save_path)

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)