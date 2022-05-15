from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp
import yaml
import torch
from cmd_parser import parse_config
from data_parser import create_dataset
import mmpose

def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, **args)
    for idx, data in enumerate(dataset_obj):

        img = data['img']
        fn = data['fn']
        keypoints = data['keypoints']
        print('Processing: {}'.format(data['img_path']))
        print(keypoints.shape)
        exit()
        img_path = data['img_path']
        img_ext = img_path.split('.')[-1]

        curr_result_folder = osp.join(result_folder, fn)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_mesh_folder = osp.join(mesh_folder, fn)
        if not osp.exists(curr_mesh_folder):
            os.makedirs(curr_mesh_folder)

        for person_id in range(1):
            pass

if __name__ == "__main__":
    args = parse_config()
    main(**args)