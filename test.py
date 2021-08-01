# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function
import open3d as o3d

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *


def depth_to_pc(z):
    fx, fy = 388, 388
    cx, cy = 322.0336, 236.3574
    px, py = np.meshgrid(np.arange(640), np.arange(480))
    px, py = px.astype(float), py.astype(float)
    x = ((px - cx) / fx) * z 
    y = ((py - cy) / fy) * z 
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    valid_idx = (z > 0.3) & (z < 3) & (y < 0.8)
    x = x[valid_idx]
    y = y[valid_idx]
    z = z[valid_idx]


    # xyz = np.stack([x, y, z], -1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.002)
    # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=20,
    #                                             radius=0.005)
    # x = x[ind]
    # y = y[ind]
    # z = z[ind]

    # random_idx = np.random.choice(x.shape[0], int(x.shape[0]/2), replace=False)
    # x = x[random_idx]
    # y = y[random_idx]
    # z = z[random_idx]
    return x, y ,z


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            import pdb;pdb.set_trace()
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = os.path.join('results', args.model_name)
    os.makedirs(save_name, exist_ok=True)
    
    
    for s in tqdm(range(num_test_samples)):
        if args.dataset == 'kitti':
            date_drive = lines[s].split('/')[1]
            filename_pred_png = save_name + '/raw/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[
                -1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
        elif args.dataset == 'kitti_benchmark':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
        elif args.dataset == 'custom':
            scene = lines[s].split()[0].split('/')[2]
            filename_pred_png = save_name + '/pred/' + scene + '/' + lines[s].split()[0].split('/')[-1]
            filename_pred_cmap_png = save_name + '/pred_cmap/' + scene + '/' + lines[s].split()[0].split('/')[-1]
            filename_gt_png = save_name + '/gt/' + scene + '/' + lines[s].split()[0].split('/')[-1]
            filename_gt_cmap_png = save_name + '/gt_cmap/' + scene + '/' + lines[s].split()[0].split('/')[-1]
            filename_image_png = save_name + '/rgb/' + scene + '/' + lines[s].split()[0].split('/')[-1]
            filename_compare = save_name + '/compare/' + scene + '/' + lines[s].split()[0].split('/')[-1]
        else:
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]
        
        rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)
        elif args.dataset == 'custom':
            gt_path = os.path.join(args.data_path + '/' + lines[s].split()[1])
            gt = Image.open(gt_path)
            gt_cmap_path = './data/custom_data/depth_vis/' + '/'.join(lines[s].split()[1].split('/')[-3:])
            gt_cmap = cv2.imread(gt_cmap_path)
        
        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]
        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        elif args.dataset == 'custom':
            pred_depth_scaled = pred_depth * 1000.0
        else:
            pred_depth_scaled = pred_depth * 1000.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(pred_depth_scaled.astype(np.int32), alpha=0.1), cv2.COLORMAP_JET)

        os.makedirs("/".join(filename_pred_png.split('/')[:-1]), exist_ok=True)
        os.makedirs("/".join(filename_pred_cmap_png.split('/')[:-1]), exist_ok=True)
        os.makedirs("/".join(filename_gt_png.split('/')[:-1]), exist_ok=True)
        os.makedirs("/".join(filename_gt_cmap_png.split('/')[:-1]), exist_ok=True)
        os.makedirs("/".join(filename_image_png.split('/')[:-1]), exist_ok=True)
        os.makedirs("/".join(filename_compare.split('/')[:-1]), exist_ok=True)

        Image.fromarray(pred_depth_scaled).save(filename_pred_png)
        gt.save(filename_gt_png)
        cv2.imwrite(filename_pred_cmap_png, depth_colormap)
        cv2.imwrite(filename_gt_cmap_png, gt_cmap)
        cv2.imwrite(filename_image_png, image)
        

        # compare
        compare = np.concatenate([image, gt_cmap, depth_colormap], 1)
        cv2.imwrite(filename_compare, compare)

        # BEV
        gt = np.asanyarray(gt).astype(float) / 1000.
        pred = pred_depth_scaled / 1000.

        gt_x, gt_y, gt_z = depth_to_pc(gt)
        pred_x, pred_y, pred_z = depth_to_pc(pred)

        fig = plt.figure()
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 2, 3)
        ax2 = fig.add_subplot(2, 2, 4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax0.imshow(image)
        ax0.axes.xaxis.set_visible(False)
        ax0.axes.yaxis.set_visible(False)
        ax1.scatter(gt_x, gt_z, s=0.01)
        ax2.scatter(pred_x, pred_z, s=0.01)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(0, 3)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(0, 3)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1.set_title('Ground Truth')
        ax2.set_title('Prediction')
        folder, filename = lines[s].split()[1].split('/')[-2:]
        bev_path = os.path.join(save_name, 'bev', folder)
        os.makedirs(bev_path, exist_ok=True)
        plt.savefig(os.path.join(bev_path, filename))
        plt.close()




        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
                pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')

    # images to video (compare)
    img_path = os.path.join(save_name, 'compare')
    video_path = os.path.join(save_name, 'video_depth')
    image_folders = os.listdir(img_path)
    for image_folder in image_folders:
        image_folder = os.path.join(img_path, image_folder)
        video_name = image_folder.split('/')[-1] + '.mp4'
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        os.makedirs(video_path, exist_ok=True)
        video_name = os.path.join(video_path, video_name)
        video = cv2.VideoWriter(video_name, 0, 10, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
    
    # images to video (BEV)
    img_path = os.path.join(save_name, 'bev')
    video_path = os.path.join(save_name, 'video_bev')
    image_folders = os.listdir(img_path)
    for image_folder in image_folders:
        image_folder = os.path.join(img_path, image_folder)
        video_name = image_folder.split('/')[-1] + '.mp4'
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        os.makedirs(video_path, exist_ok=True)
        video_name = os.path.join(video_path, video_name)
        video = cv2.VideoWriter(video_name, 0, 10, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
    
    
    return


if __name__ == '__main__':
    test(args)
