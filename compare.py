import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Save figures to compare gt and pred')
parser.add_argument('result')

args = parser.parse_args()

folders = os.listdir(os.path.join(args.result, 'gt'))
for _, folder in enumerate(tqdm(folders)):
    filenames = os.listdir(os.path.join(args.result, 'gt', folder))
    for _, filename in enumerate(tqdm(filenames)):
        # save_path = os.path.join(args.result, 'compare', folder)
        save_path = os.path.join(args.result, 'compare_3d', folder)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, filename)




        img_path = os.path.join(args.result, 'rgb', folder, filename)
        # gt_path = os.path.join(args.result, 'gt_cmap', folder, filename)
        gt_path = os.path.join(args.result, 'gt_3d', folder, filename)
        # pred_path = os.path.join(args.result, 'pred_cmap', folder, filename)
        pred_path = os.path.join(args.result, 'pred_3d', folder, filename)

        img = Image.open(img_path)
        gt = Image.open(gt_path)
        pred = Image.open(pred_path)
        img = np.asanyarray(img)
        gt = np.asanyarray(gt)
        pred = np.asanyarray(pred)

        # compare = np.concatenate([img, gt, pred], 1)
        compare = np.concatenate([gt, pred], 1)
        im = Image.fromarray(compare)
        im.save(save_path)
