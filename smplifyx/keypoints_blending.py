import mmcv
import math
from mmcv.visualization.image import imshow
from mmcv.image import imwrite
from mmpose.core import imshow_bboxes
import os
from os.path import exists, join, basename, splitext
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import json
from collections import namedtuple

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)

def imshow_keypoints_modified(img,
                     pose_result,
                     skeleton=None,
                     kpts_score_thr=[0.3] * 136,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score < kpts_score_thr[kid] or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpts_score_thr[sk[0]]
                        or kpts[sk[1], 2] < kpts_score_thr[sk[1]]
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return 

def show_result_modified(img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):

        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints_modified(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

def vis_pose_result_body_25(model,
                    img,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    dataset_info=None,
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    # get dataset info
    palette_body = np.array([[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], 
        [85, 255, 0], [0, 255, 0], [255, 0, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
        [0, 0, 255], [255, 0, 170], [170, 0, 255], [255, 0, 255], [85, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], 
        [0, 255, 255], [0, 255, 255], [0, 255, 255]])
    body_skeleton = np.array([[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], 
    [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]])
    body_link_color = palette_body[body_skeleton[:,1]]

    palette_hand = np.array([[100, 100, 100], [100, 0, 0], [150, 0, 0], [200, 0, 0], [255, 0, 0], [100, 100, 0], 
        [150, 150, 0], [200, 200, 0], [255, 255, 0], [0, 100, 50], [0, 150, 75], [0, 200, 100], [0, 255, 125], 
        [0, 50, 100], [0, 75, 150], [0, 100, 200], [0, 125, 255], [100, 0, 100], [150, 0, 150], [200, 0, 200], [255, 0, 255]])
    hand_skeleton = np.array([[0,1], [1,2], [2,3], [3,4], [0,5], [5,6], [6,7], [7,8], [0,9], [9,10], [10,11], 
        [11,12], [0,13], [13,14], [14,15], [15,16], [0,17], [17,18], [18,19], [19,20]])
    hand_skeleton_left = hand_skeleton + 25
    hand_skeleton_right = hand_skeleton_left + 21
    hand_link_color = palette_hand[hand_skeleton[:,1]]
    pose_link_color = np.concatenate((body_link_color, hand_link_color, hand_link_color), axis=0)
    skeleton = np.concatenate((body_skeleton, hand_skeleton_left, hand_skeleton_right), axis=0)
    palette_face = np.array([[255, 255, 255]] * 68)
    pose_kpt_color = np.concatenate((palette_body, palette_hand, palette_hand, palette_face), axis=0)
    
    img = show_result_modified(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img

def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False, orders=['body', 'hands', 'face']):
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
            hands_keypoints = np.concatenate([left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:68, :]

        if orders[0] == 'body':
          curr_keypoints = body_keypoints
        elif orders[0] == 'hands':
          curr_keypoints = hands_keypoints
        else:
          curr_keypoints = face_keypoints
        
        if orders[1] == 'body':
          curr_keypoints = np.concatenate([curr_keypoints, body_keypoints])
        elif orders[1] == 'hands':
          curr_keypoints = np.concatenate([curr_keypoints, hands_keypoints])
        else:
          curr_keypoints = np.concatenate([curr_keypoints, face_keypoints])

        if orders[2] == 'body':
          curr_keypoints = np.concatenate([curr_keypoints, body_keypoints])
        elif orders[2] == 'hands':
          curr_keypoints = np.concatenate([curr_keypoints, hands_keypoints])
        else:
          curr_keypoints = np.concatenate([curr_keypoints, face_keypoints])

        keypoints.append(curr_keypoints)

    return keypoints[0]

def blending(IMAGES_PATH, OPENPOSE_RES_DIR, MMPOSE_RES_DIR, BLENDING_RES_DIR):
    # Load keypoints
    openpose_files = {}
    mmpose_files = {}
    for fn in glob.glob(os.path.join(IMAGES_PATH, '*')):
        img_name = fn.split('/')[-1].split('.')[0]
        openpose_fn = os.path.join(OPENPOSE_RES_DIR, img_name + '_keypoints.json')
        openpose_files[img_name] = read_keypoints(openpose_fn, use_hands=True, use_face=True, use_face_contour=True)
        mmpose_fn = os.path.join(MMPOSE_RES_DIR, img_name + '_mmpose.json')
        mmpose_files[img_name] = read_keypoints(mmpose_fn, use_hands=True, use_face=True, use_face_contour=True)

    # Matching keypoints for different formats
    pairs = {"Nose": {'MMPose': 0, 'OpenPose': 0},
          "LEye": {'MMPose': 1, 'OpenPose': 16},
          "REye": {'MMPose': 2, 'OpenPose': 15},
          "LEar": {'MMPose': 3, 'OpenPose': 18},
          "REar": {'MMPose': 4, 'OpenPose': 17},
          "LShoulder": {'MMPose': 5, 'OpenPose': 5},
          "RShoulder": {'MMPose': 6, 'OpenPose': 2},
          "LElbow": {'MMPose': 7, 'OpenPose': 6},
          "RElbow": {'MMPose': 8, 'OpenPose': 3},
          "LWrist": {'MMPose': 9, 'OpenPose': 7},
          "RWrist": {'MMPose': 10, 'OpenPose': 4},
          "LHip": {'MMPose': 11, 'OpenPose': 12},
          "RHip": {'MMPose': 12, 'OpenPose': 9},
          "LKnee": {'MMPose': 13, 'OpenPose': 13},
          "RKnee": {'MMPose': 14, 'OpenPose': 10},
          "LAnkle": {'MMPose': 15, 'OpenPose': 14},
          "RAnkle": {'MMPose': 16, 'OpenPose': 11},
          "Neck": {'MMPose': 18, 'OpenPose': 1},
          "Hip": {'MMPose': 19, 'OpenPose': 8},
          "LBigToe": {'MMPose': 20, 'OpenPose': 19},
          "RBigToe": {'MMPose': 21, 'OpenPose': 22},
          "LSmallToe": {'MMPose': 22, 'OpenPose': 20},
          "RSmallToe":{'MMPose': 23, 'OpenPose': 23},
          "LHeel": {'MMPose': 24, 'OpenPose': 21},
          "RHeel": {'MMPose': 25, 'OpenPose': 24}}

    openpose_pose_len = 25
    mmpose_pose_len = 26
    for i in range(21):
        key = 'left_hand_' + str(i+1)
        pairs[key] = {}
        pairs[key]['OpenPose'] = openpose_pose_len + i
        pairs[key]['MMPose'] = mmpose_pose_len + i
    for i in range(21):
        key = 'right_hand_' + str(i+1)
        pairs[key] = {}
        pairs[key]['OpenPose'] = openpose_pose_len + 21 + i
        pairs[key]['MMPose'] = mmpose_pose_len + 21 + i
    for i in range(68):
        key = 'face_' + str(i+1)
        pairs[key] = {}
        pairs[key]['OpenPose'] = openpose_pose_len + 42 + i
        pairs[key]['MMPose'] = mmpose_pose_len + 42 + i

    with open('/content/heuristics/openpose_means.json', 'r') as f:
        openpose_means = json.load(f)
    with open('/content/heuristics/openpose_stds.json', 'r') as f:
        openpose_stds = json.load(f)
    with open('/content/heuristics/mmpose_means.json', 'r') as f:
        mmpose_means = json.load(f)
    with open('/content/heuristics/mmpose_stds.json', 'r') as f:
        mmpose_stds = json.load(f)

    # Blending
    for fn in glob.glob(os.path.join(IMAGES_PATH, '*')):
        img_name = fn.split('/')[-1].split('.')[0]
        blended_current = np.zeros((135, 3))
        for key in pairs:
            if 'face' in key:
                openpose_conf = openpose_files[img_name][pairs[key]['OpenPose']][2]
                openpose_conf = np.clip(openpose_conf, 0, 1)
                blended_current[pairs[key]['OpenPose']][0] = openpose_files[img_name][pairs[key]['OpenPose']][0]
                blended_current[pairs[key]['OpenPose']][1] = openpose_files[img_name][pairs[key]['OpenPose']][1]
                blended_current[pairs[key]['OpenPose']][2] = openpose_conf
            else:
                openpose_conf = -1
                if 'OpenPose' in pairs[key]:
                    openpose_conf = openpose_files[img_name][pairs[key]['OpenPose']][2]
                    openpose_conf = np.clip(openpose_conf, 0, 1)
                mmpose_conf = -1
                if 'MMPose' in pairs[key]:
                    mmpose_conf = mmpose_files[img_name][pairs[key]['MMPose']][2]
                    mmpose_conf = (mmpose_conf - mmpose_means[key]) / mmpose_stds[key]
                    mmpose_conf = mmpose_conf * openpose_stds[key] + openpose_means[key]
                    mmpose_conf = np.clip(mmpose_conf, 0, 1)
        
                if mmpose_conf > openpose_conf:
                    blended_current[pairs[key]['OpenPose']][0] = mmpose_files[img_name][pairs[key]['MMPose']][0]
                    blended_current[pairs[key]['OpenPose']][1] = mmpose_files[img_name][pairs[key]['MMPose']][1]
                    blended_current[pairs[key]['OpenPose']][2] = mmpose_conf
                else:
                    blended_current[pairs[key]['OpenPose']][0] = openpose_files[img_name][pairs[key]['OpenPose']][0]
                    blended_current[pairs[key]['OpenPose']][1] = openpose_files[img_name][pairs[key]['OpenPose']][1]
                    blended_current[pairs[key]['OpenPose']][2] = openpose_conf

    blended_current = blended_current.flatten().tolist()
    param_dict = {}
    param_dict['people'] = [{"person_id":[-1]}]
    param_dict['people'][0]['pose_keypoints_2d'] = blended_current[:25*3]
    param_dict['people'][0]['hand_left_keypoints_2d'] = blended_current[25*3:46*3]
    param_dict['people'][0]['hand_right_keypoints_2d'] = blended_current[46*3:67*3]
    param_dict['people'][0]['face_keypoints_2d'] = blended_current[67*3:]
    with open(os.path.join(BLENDING_RES_DIR, img_name + '_blended.json'), 'w') as outfile:
        json.dump(param_dict, outfile, indent=2)