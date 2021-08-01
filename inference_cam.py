from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

import torch
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

from bts_dataloader import BtsDataLoader
from bts import BtsModel


def depth_to_pc(z):
    """
    Depth map을 받아 point cloud 좌표 x, y, z를 반환
    args
        z: ndarray 
    return
        (x, y, z):
    """
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
    x_ = x[valid_idx]
    y_ = y[valid_idx]
    z_ = z[valid_idx]

    # kitti lidar coordinate
    x = z_
    y = -x_
    z = y_

    return (x, y ,z)

def count_points_in_box(x, y, z, point, offset):
    idx = np.where(
        ((point[0] - offset) < x) &\
        ((point[0] + offset) > x) &\
        ((point[1] - offset) < y) &\
        ((point[1] + offset) > y) &\
        ((point[2] - offset) < z) &\
        ((point[2] + offset) > z))[0]
    num_points_in_box = idx.shape[0]
    
    return num_points_in_box

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

    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)

    standard_num_points = 2
    points_interest = ((1, 0, 0),
                       (2, 2, 2))
    offset = 0.1

    while True:
        warning = False
        start_time = time.time()
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        image = np.asanyarray(frame.get_data()).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        focal = torch.tensor([388.], dtype=torch.float64)

        with torch.no_grad():
            image = Variable(image.cuda())
            focal = Variable(focal.cuda())
            depth = model(image, focal)[-1].cpu().numpy()

        x, y, z = depth_to_pc(depth)

        for point_interest in points_interest:
            num_points = count_points_in_box(x, y, z, point_interest, offset)
            if num_points > standard_num_points:
                warning = True
                break

        end_time = time.time()
        print(f'Warning: {warning}', end='')
        print(f'\tnum_points:{num_points}', end=''),
        print(f'\tFPS: {1 / (end_time - start_time)}')
        

if __name__ == '__main__':
    test(args)
