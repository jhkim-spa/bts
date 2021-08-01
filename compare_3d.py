import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Save figures to compare gt and pred')
parser.add_argument('model')

args = parser.parse_args()
result_folder = f'results/{args.model}'

folders = os.listdir(os.path.join(result_folder, 'gt'))
for _, folder in enumerate(tqdm(folders)):
    filenames = os.listdir(os.path.join(result_folder, 'gt', folder))
    for _, filename in enumerate(tqdm(filenames)):
        # save_path = os.path.join(args.result, 'compare', folder)
        save_path = os.path.join(result_folder, 'compare_3d', folder)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, filename)




        img_path = os.path.join(result_folder, 'rgb', folder, filename)
        gt_front_path = os.path.join(result_folder, 'gt_3d_front', folder, filename)
        gt_bev_path = os.path.join(result_folder, 'gt_3d_bev', folder, filename)
        pred_front_path = os.path.join(result_folder, 'pred_3d_front', folder, filename)
        pred_bev_path = os.path.join(result_folder, 'pred_3d_bev', folder, filename)

        img = Image.open(img_path)
        gt_front = Image.open(gt_front_path)
        gt_bev = Image.open(gt_bev_path)
        pred_front = Image.open(pred_front_path)
        pred_bev = Image.open(pred_bev_path)
        img = np.asanyarray(img)
        gt_front = np.asanyarray(gt_front)
        gt_bev = np.asanyarray(gt_bev)
        pred_front = np.asanyarray(pred_front)
        pred_bev = np.asanyarray(pred_bev)

        # compare = np.concatenate([img, gt, pred], 1)
        pad = np.ones((480, 1300, 3), dtype=np.uint8) * 255
        img = np.concatenate([pad, img, pad], 1)
        front = np.concatenate([gt_front, pred_front], 0)
        bev = np.concatenate([gt_bev, pred_bev], 0)
        compare = np.concatenate([front[:, 300:, :], bev[:, 300:, :]], 1)
        compare = np.concatenate([img, compare], 0)
        im = Image.fromarray(compare)
        im.save(save_path)
