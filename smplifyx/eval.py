import glob
import os
import pickle
import re
from collections import defaultdict
import numpy as np
from plyfile import PlyData
import tqdm
from camera import PerspectiveCameraCroppedEHFGT
import torch
from utils import ProcrustesAlignmentMPJPE, PelvisAlignmentMPJPE

def compute_v2v(vertices_fitted, vertices_target, alignments, vids=None):
    if torch.is_tensor(vertices_fitted):
        vertices_fitted = vertices_fitted.detach().cpu().numpy()
    if torch.is_tensor(vertices_target):
        vertices_target = vertices_target.detach().cpu().numpy()

    gt_vertices = vertices_target
    model_vertices = vertices_fitted

    if vids is not None:
        gt_vertices = vertices_target[:, vids]
        model_vertices = vertices_fitted[:, vids]

    v2v_err = {}
    fscores = {}
    for alignment_name, alignment in alignments.items():
        v2v_err[alignment_name] = []
        fscores[alignment_name] = defaultdict(lambda: [])

        for bidx in range(gt_vertices.shape[0]):
            align_out = alignment(
                model_vertices[bidx], gt_vertices[bidx])
            v2v_err[alignment_name].append(align_out['point'])
            for thresh, val in align_out['fscore'].items():
                fscores[alignment_name][thresh].append(
                    val['fscore'].copy())

        v2v_err[alignment_name] = np.stack(v2v_err[alignment_name])
        for thresh in fscores[alignment_name]:
            fscores[alignment_name][thresh] = np.stack(fscores[alignment_name][thresh])

    return {'point': v2v_err, 'fscore': fscores}

def load(root_dir, gt=False):
    vertices = {}
    print('Loading {}'.format(('ground truth meshes' if gt else 'fitted meshes') + ':'))
    for f in tqdm.tqdm(glob.glob(os.path.join(root_dir, '**/*.ply'), recursive=True)):
        if gt:
            img_name = re.split(r'/|\\', f)[-1].split('_')[0] + '_cropped'
        else:
            img_name = re.split(r'/|\\', f)[-2]
        dict = PlyData.read(f)
        body_vertices = dict.elements[0].data
        body_vertices = torch.tensor([list(vertex) for vertex in body_vertices]).unsqueeze(0)
        vertices[img_name] = body_vertices
    return vertices

def get_indices_in_bound(projected_vertices):
    idx_in_bound = []
    for idx in range(len(projected_vertices[0])):
        if projected_vertices[:, idx, 0] >= 0 and projected_vertices[:, idx, 0] < 800 \
                and projected_vertices[:, idx, 1] >= 0 and projected_vertices[:, idx, 1] < 600:
            idx_in_bound.append(idx)
    return idx_in_bound

alignments = {'procrustes': ProcrustesAlignmentMPJPE(),
              'pelvis': PelvisAlignmentMPJPE()}

with open(r"MANO_SMPLX_vertex_ids.pkl", "rb") as input_file:
    d = pickle.load(input_file)
    left_hand_vertices_id = d['left_hand']
    right_hand_vertices_id = d['right_hand']
face_vertices_id = np.load('SMPL-X__FLAME_vertex_ids.npy')
body_vertices_id = np.load('SMPL-X__BODY_vertex_ids.npy')

j14_regressor_path_smplx = 'SMPLX_to_J14.pkl'
with open(j14_regressor_path_smplx, 'rb') as f:
    J14_regressor_smplx = pickle.load(f, encoding='latin1')

gt_vertices_all = load(root_dir='../EHF', gt=True)
fitted_vertices_all = load(root_dir='../smplifyx_results_combined_prior_blend_body_openpose_face_f5000_sshq_norm_body_thr_only_higher_jaw_prior_weights', gt=False)
v2v_whole_observed_all = {}
v2v_body_all = {}
v2v_face_all = {}
v2v_left_hand_all = {}
v2v_right_hand_all = {}
mpjpe_14_all = {}
print('Evaluating: ')
for key in tqdm.tqdm(gt_vertices_all.keys()):
    gt_vertices = gt_vertices_all[key]
    gt_joints = np.einsum(
        'jv,bvm->bjm', J14_regressor_smplx, gt_vertices.detach().numpy())
    fitted_vertices = fitted_vertices_all[key]
    fitted_joints = np.einsum(
        'jv,bvm->bjm', J14_regressor_smplx, fitted_vertices.detach().numpy())
    with open('../EHF_bbox/' + key + '.txt', 'r') as f:
        xmin, xmax, ymin, ymax = [float(i) for i in f.read().split(' ')]
    camera_gt = PerspectiveCameraCroppedEHFGT(xmin=xmin, ymin=ymin)
    gt_vertices_projected = camera_gt(gt_vertices)
    in_bound_indices = get_indices_in_bound(gt_vertices_projected)
    in_bound_indices_body = list(set(in_bound_indices) & set(body_vertices_id))
    in_bound_indices_face = list(set(in_bound_indices) & set(face_vertices_id))
    in_bound_indices_left_hand = list(set(in_bound_indices) & set(left_hand_vertices_id))
    in_bound_indices_right_hand = list(set(in_bound_indices) & set(right_hand_vertices_id))
    gt_joints_projected = camera_gt(torch.tensor(gt_joints).float())
    in_bound_joints = get_indices_in_bound(gt_joints_projected)
    gt_whole_observed = gt_vertices[:, in_bound_indices, :]
    gt_body = gt_vertices[:, in_bound_indices_body, :]
    gt_face = gt_vertices[:, in_bound_indices_face, :]
    gt_left_hand = gt_vertices[:, in_bound_indices_left_hand, :]
    gt_right_hand = gt_vertices[:, in_bound_indices_right_hand, :]
    gt_joints_observed = gt_joints[:, in_bound_joints, :]

    fitted_whole_observed = fitted_vertices[:, in_bound_indices, :]
    fitted_body = fitted_vertices[:, in_bound_indices_body, :]
    fitted_face = fitted_vertices[:, in_bound_indices_face, :]
    fitted_left_hand = fitted_vertices[:, in_bound_indices_left_hand, :]
    fitted_right_hand = fitted_vertices[:, in_bound_indices_right_hand, :]
    fitted_joints_observed = fitted_joints[:, in_bound_joints, :]

    v2v_whole_observed = compute_v2v(fitted_whole_observed, gt_whole_observed, alignments)
    v2v_whole_observed_all[key] = v2v_whole_observed['point']['procrustes'].mean()
    v2v_body = compute_v2v(fitted_body, gt_body, alignments)
    v2v_body_all[key] = v2v_body['point']['procrustes'].mean()
    v2v_face = compute_v2v(fitted_face, gt_face, alignments)
    v2v_face_all[key] = v2v_face['point']['procrustes'].mean()
    v2v_left_hand = None
    v2v_right_hand = None
    if len(in_bound_indices_left_hand) > 0:
        v2v_left_hand = compute_v2v(fitted_left_hand, gt_left_hand, alignments)
        v2v_left_hand_all[key] = v2v_left_hand['point']['procrustes'].mean()
    if len(in_bound_indices_right_hand) > 0:
        v2v_right_hand = compute_v2v(fitted_right_hand, gt_right_hand, alignments)
        v2v_right_hand_all[key] = v2v_right_hand['point']['procrustes'].mean()
    mpjpe_14 = compute_v2v(fitted_joints_observed, gt_joints_observed, alignments)
    mpjpe_14_all[key] = mpjpe_14['point']['procrustes'].mean()

print('All: {:.4f}, Body: {:.4f}, Face: {:.4f}, Left Hand: {:.4f}, Right Hand: {:.4f}, MPJPE-14: {:.4f}'.format(
    1000*np.mean(list(v2v_whole_observed_all.values())),
    1000*np.mean(list(v2v_body_all.values())),
    1000*np.mean(list(v2v_face_all.values())),
    1000*np.mean(list(v2v_left_hand_all.values())),
    1000*np.mean(list(v2v_right_hand_all.values())),
    1000*np.mean(list(mpjpe_14_all.values()))))