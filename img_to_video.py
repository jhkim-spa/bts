import cv2
import os
import glob
import argparse
import pdb


parser = argparse.ArgumentParser(description='Generate video files from images')
parser.add_argument('img_path')
parser.add_argument('video_path')

args = parser.parse_args()

image_folders = os.listdir(args.img_path)
for image_folder in image_folders:
    image_folder = os.path.join(args.img_path, image_folder)
    video_name = image_folder.split('/')[-1] + '.avi'
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video_path = os.path.join(args.video_path, video_name)
    os.makedirs(args.video_path, exist_ok=True)
    video = cv2.VideoWriter(video_path, 0, 3, (width,height))
    video_folder = os.path.join(args.video_path, image_folder.split('/')[-1])
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()