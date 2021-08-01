from PIL import Image
import numpy as np
import glob
import open3d as o3d
from open3d import geometry
import os
from utils.compare_3d_utils import depth_to_pc, scatter_3d, visualize
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--plotly', action='store_true')

args = parser.parse_args()

window_made = False

camera_dict = dict(
    outdoor4 = dict(
        front = {"front": [ 0.050501787897276829, -0.52570778609624835, -0.84916482090166667 ],
                "lookat": [ -0.487971344481411, -25.997708747160416, 13.178550565517178 ],
                "up": [ 0.033366379741624194, -0.84888700869468015, 0.52752017134166151 ],
                "zoom": 0.10000000000000001},
        bev = {"front" : [ 0.023110020160663072, -0.99392636875572671, -0.10759321753822966 ],
            "lookat" : [ 2.7075857639198007, -32.583199011414585, 43.587691612460297 ],
            "up" : [ -0.035191561163714677, -0.10836403218062425, 0.99348819346402761 ],
            "zoom" : 0.12000000000000001},
        x = np.array([0.266, 0.4]),
        y = np.array([0.201, 0.1]),
        z = np.array([0.712, 0.872])),
    parking3 = dict(
        front = {"front": [ -0.04810098369685168, -0.48531027359644346, -0.87301788853902673 ],
                "lookat": [ 1.2977373278522735, -9.3907821962656506, 10.93462517771299 ],
                "up": [ 0.21125537801823679, -0.85920022207895907, 0.46598942438411572 ],
                "zoom": 0.080000000000000002},
        bev = {"front" : [ 0.041095469531816371, -0.99895401578472498, -0.020050853635969555 ],
            "lookat" : [ 0.39973802559975352, -10.698656558838893, 29.770899144098998 ],
            "up" : [ 0.23957793410268155, -0.0096304632452983499, 0.97082937103734768 ],
            "zoom" : 0.12000000000000001},
        x = np.array([0.192, 0.294]),
        y = np.array([0.324, 0.223]),
        z = np.array([0.678, 0.81])))


view_list = ['front', 'bev']
scene_list = ['outdoor4', 'parking3']
source_list = ['gt', 'pred']
scale = 50

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.set_full_screen(True)
vis.get_render_option().point_size = 0.01
for scene in scene_list:
    for view in view_list:
        for source in source_list:

            camera = camera_dict[scene]
            front = camera[view]['front']
            lookat = camera[view]['lookat']
            up = camera[view]['up']
            zoom = camera[view]['zoom']
            x, y , z = camera['x'], camera['y'], camera['z']
            pi = 0.1745 # 10도
            x_ = (x * math.cos(pi) - z * math.sin(pi)).reshape(-1, 1)
            z_ = (x * math.sin(pi) + z * math.cos(pi)).reshape(-1, 1)
            y_ = y.reshape(-1, 1)
            centers = np.concatenate([x_, y_, z_], 1)
            offset = 0.05
            size = np.array([offset, offset, offset]).reshape(3, 1)
            centers *= scale
            size *= scale
            pcd = geometry.PointCloud()

            depth_list = glob.glob(f'results/{args.model}/{source}/{scene}/*.png')
            out_path = f'results/{args.model}/{source}_3d_{view}'

            for depth_path in depth_list:
                depth = Image.open(depth_path)
                depth = np.asanyarray(depth)
                x, y , z = depth_to_pc(depth)

                valid_idx = (z > 0.2) & (z < 2) & (y > -1) & (0.78 > y) & (-1 < x) & (x < 1)
                x = x[valid_idx]
                y = y[valid_idx]
                z = z[valid_idx]

                # for same view point
                x = np.concatenate([x, np.array([-1, -1, -1, -1, 1, 1, 1, 1])], dtype=x.dtype)
                y = np.concatenate([y, np.array([1, -1, 1, -1, 1, -1, 1, -1])], dtype=x.dtype)
                z = np.concatenate([z, np.array([0, 0, 3, 3, 3, 3, 0, 0])], dtype=x.dtype)


                if args.plotly:
                    scatter_3d(x, y, z)
                points = np.stack([x, y, z]).T
                points_colors = np.tile(np.array([0.7, 0.7, 0.7]), (points.shape[0], 1))
                pcd.points = o3d.utility.Vector3dVector(points * scale)

                line_set_list = []
                for i in range(centers.shape[0]):
                    center = centers[i]
                    center = center.reshape(3, 1)
                    angle = np.zeros(3)
                    rot_mat = geometry.get_rotation_matrix_from_xyz(angle)
                    box3d = geometry.OrientedBoundingBox(center, rot_mat, size)
                    box3d.color = np.array([1, 0, 0])
                    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)

                    indices = box3d.get_point_indices_within_bounding_box(pcd.points)
                    points_colors[indices] = np.array([1, 0, 0])
                    vis.add_geometry(line_set)
                pcd.colors = o3d.utility.Vector3dVector(points_colors)
                vis.add_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                filename = depth_path.split('/')[-1]
                save_path = os.path.join(out_path, scene)
                save_path = os.path.join(save_path, filename)
                view_ctl = vis.get_view_control()
                view_ctl.set_lookat(lookat)
                view_ctl.set_front(front)
                view_ctl.set_up(up)
                view_ctl.set_zoom(zoom)
                vis.run()
                vis.capture_screen_image(save_path, True)
            vis.clear_geometries()

