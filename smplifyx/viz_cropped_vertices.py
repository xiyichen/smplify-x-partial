import os

import numpy as np
from PIL import Image, ImageDraw

from dataloaders.EHF import EHF
import tqdm
from smplifyx.camera import PerspectiveCamera, PerspectiveCameraCroppedEHFGT
from smplifyx.cmd_parser import parse_config
import torch
from evaluation import Evaluator
from smplifyx.utils import ProcrustesAlignmentMPJPE, PelvisAlignmentMPJPE
from dataloaders.PIXIE import PIXIE

device = torch.device('cpu')
args = parse_config()
ehf_gt = EHF(root_dir='../EHF', suffix='.ply', device=device, method='gt', **args)
ehf_smplifyx = EHF(root_dir='../smplifyx_pixie_results_w_vertices/', suffix='.ply', method='smplify-pixie', device=device, **args)
v2vs = []
evaluator = Evaluator(exp_cfg=args)
alignments = {'procrustes': ProcrustesAlignmentMPJPE(),
              'pelvis': PelvisAlignmentMPJPE(),
              }
v2vs = []
gender = 'male'
pixie = PIXIE(root_dir='../PIXIE/TestSamples/body/results',
              smplx_path=os.path.join(args.get('model_folder'), 'smplx', 'SMPLX_' + gender.upper() + '.npz'),
              device=device)



def get_indices_in_bound(projected_vertices):
    idx_in_bound = []

    for idx in range(len(projected_vertices[0])):
        if projected_vertices[:, idx, 0] >= 0 and projected_vertices[:, idx, 0] < 800 \
                and projected_vertices[:, idx, 1] >= 0 and projected_vertices[:, idx, 1] < 600:
            idx_in_bound.append(idx)
    return idx_in_bound

for idx in tqdm.tqdm(range(len(ehf_gt))):
    body_vertices, camera_transl, camera_center, img_name, focal_length = ehf_smplifyx.__getitem__(idx)
    body_targets, _, _, _, _ = ehf_gt.__getitem__(idx)
    camera = PerspectiveCamera()
    with torch.no_grad():
        camera.translation[:] = torch.tensor(camera_transl, device=device).float()
        camera.center[:] = torch.tensor(camera_center, device=device).float()
        camera.focal_length_x = torch.tensor(focal_length)
        camera.focal_length_y = torch.tensor(focal_length)
    with open('../EHF_bbox/' + img_name + '.txt', 'r') as f:
        xmin, xmax, ymin, ymax = [float(i) for i in f.read().split(' ')]
    camera_gt = PerspectiveCameraCroppedEHFGT(xmin=xmin, ymin=ymin)
    # print(camera_gt.center)
    camera_gt_center = camera_gt.center.detach().clone().numpy()[0]
    with torch.no_grad():
        camera_gt.center[:] = torch.tensor([[camera_gt_center[0], camera_gt_center[1]]], device=device).float()
    # exit()

    # body_vertices[:,:,1] *= -1
    # body_vertices[:,:,1] += -1.2246468e-16
    # body_vertices[:,:,2] *= -1
    # body_vertices[:,:,2] += 1.2246468e-16

    projected_vertices = camera(body_vertices)
    projected_gt = camera_gt(body_targets)
    # print(camera.translation, camera.center, camera.focal_length_x, camera.focal_length_y)
    # print(projected_vertices.shape)
    # print(projected_vertices[:,:,0].min(), projected_vertices[:,:,0].max())
    # print(projected_vertices[:,:,1].min(), projected_vertices[:,:,1].max())




    # gt_in_bound = body_targets[:,idx_in_bound,:]
    in_bound_indices = get_indices_in_bound(projected_gt)
    gt_2d_vertices_in_bound = projected_gt[:, in_bound_indices, :]
    gt_3d_vertices_in_bound = body_targets[:, in_bound_indices, :]


    fitted_2d_vertices_in_bound = projected_vertices[:, in_bound_indices,:]
    fitted_3d_vertices_in_bound = body_vertices[:, in_bound_indices,:]
    pixie_3d_vertices, pixie_2d_vertices, _ = pixie.__getitem__(idx)
    pixie_3d_vertices_in_bound = pixie_3d_vertices[:, in_bound_indices, :]
    # exit()
    v2v_output = evaluator.compute_v2v(body_vertices, body_targets, alignments)
    v2vs.append(v2v_output['point']['procrustes'].mean())
    print(v2v_output['point']['procrustes'].mean())


    # img = Image.open('C:\\Users\\xiyi\\projects\\semester_project\\smplify-x\\data\\images\\' + img_name + '.jpg')
    # draw = ImageDraw.Draw(img)
    # for vertex in fitted_2d_vertices_in_bound[0]:
    #     x = vertex[0]
    #     y = vertex[1]
    #     draw.point((x, y), 'red')
    # img.show()
    # if idx == 1:
    #     exit()

print('mean v2v loss:', np.array(v2vs).mean())