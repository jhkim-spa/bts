from PIL import Image
import numpy as np
import glob
import open3d as o3d
from open3d import geometry
import os
from utils.compare_3d_utils import depth_to_pc, scatter_3d, visualize, LineMesh
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--plotly', action='store_true')

args = parser.parse_args()

window_made = False

camera_dict = dict(
    outdoor4 = dict(
        front = {"front": [ -0.095422956167005996, -0.12675271433723631, -0.98733388924137144 ],
                 "lookat": [ -0.052682815791452661, -0.019232212260983025, 1.1747121144300272 ],
                 "up": [ -0.035531686304387779, -0.99079415027076823, 0.13063097281117259 ],
                 "zoom": 0.3},
        bev = {"front" : [ -0.010681903675856416, -0.89386514740593936, -0.4482086513966661 ],
               "lookat" : [ 0.012341486366872607, 0.080224012063356012, 0.97481629432666672 ],
               "up" : [ 0.042851488582257874, -0.4482316538894181, 0.89288976608416881 ],
               "zoom" : 0.25},
        x = np.array([0.266, 0.497]),
        y = np.array([0.201, 0.1364]),
        z = np.array([0.712, 1.2])),
    parking3 = dict(
        front = {"front": [ -0.22171840567598744, -0.37204046763437604, -0.9013472355462645 ],
                "lookat": [ 0.12101277574962399, 0.3030084128636959, 1.5244728703844519 ],
                "up": [ 0.25727754975631867, -0.91391904624009479, 0.31394305106337445 ],
                "zoom": 0.37999999999999967},
        bev = {"front" : [ 0.10243451205516796, -0.99280407778519764, -0.062026074139049932 ],
               "lookat" : [ -0.026660891419728053, 0.70041993911166223, 0.67074046440265811 ],
               "up" : [ 0.35010955271922622, -0.022381534056698682, 0.93644133186655765 ],
               "zoom" : 0.2999999999999996},
        x = np.array([0.192, 0.294]),
        y = np.array([0.324, 0.223]),
        z = np.array([0.678, 0.81])))


view_list = ['bev', 'front']
scene_list = ['outdoor4', 'parking3']
source_list = ['gt', 'pred']
scale = 1

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
            offset = 0.06
            size = np.array([offset, offset, offset]).reshape(3, 1)
            centers *= scale
            size *= scale
            pcd = geometry.PointCloud()

            depth_list = glob.glob(f'results/{args.model}/{source}/{scene}/*.png')
            out_path = f'results/{args.model}/{source}_3d_{view}'

            for depth_path in depth_list:
                rgb_path = depth_path.split('/')
                rgb_path[2] = 'rgb'
                rgb_path = '/'.join(rgb_path)
                depth = o3d.io.read_image(depth_path)
                rgb = o3d.io.read_image(rgb_path)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb, depth, depth_scale=1000., depth_trunc=10., convert_rgb_to_intensity=False
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image,
                    o3d.camera.PinholeCameraIntrinsic(
                        640, 480, 388, 388, 322.0336, 236.3574))
                # Flip it, otherwise the pointcloud will be upside down
                # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                # x, y , z = depth_to_pc(depth)
                x = np.asarray(pcd.points)[:, 0]
                y = np.asarray(pcd.points)[:, 1]
                z = np.asarray(pcd.points)[:, 2]

                valid_idx = np.where((z > 0.2) & (z < 2) & (y > -1) & (0.78 > y) & (-1 < x) & (x < 1))[0]
                x = x[valid_idx]
                y = y[valid_idx]
                z = z[valid_idx]

                # for same view point
                x = np.concatenate([x, np.array([-1, -1, -1, -1, 1, 1, 1, 1])], dtype=x.dtype)
                y = np.concatenate([y, np.array([1, -1, 1, -1, 1, -1, 1, -1])], dtype=x.dtype)
                z = np.concatenate([z, np.array([0, 0, 3, 3, 3, 3, 0, 0])], dtype=x.dtype)
                pcd = pcd.select_by_index(valid_idx)
                col = np.zeros((8, 3), dtype=x.dtype)
                colors = np.asarray(pcd.colors)
                colors = np.concatenate([colors, col], 0)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pc = np.stack([x, y, z], 0).T
                pcd.points = o3d.utility.Vector3dVector(pc)


                if args.plotly:
                    scatter_3d(x, y, z)
                    # import pdb;pdb.set_trace()
                points = np.stack([x, y, z]).T
                points_colors = np.array([[0.7, 0.7, 0.7]]).repeat(pc.shape[0], 0)

                for i in range(centers.shape[0]):
                    center = centers[i]
                    center = center.reshape(3, 1)
                    angle = np.zeros(3)
                    rot_mat = geometry.get_rotation_matrix_from_xyz(angle)
                    box3d = geometry.OrientedBoundingBox(center, rot_mat, size)
                    box3d.color = np.array([1, 0, 0])
                    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
                    pts = np.asarray(line_set.points)
                    lines = np.asarray(line_set.lines).tolist()
                    colors = [[1, 0, 0] for i in range(len(lines))]
                    line_mesh = LineMesh(pts, lines, colors, radius=0.005)
                    line_mesh_geoms = line_mesh.cylinder_segments

                    indices = box3d.get_point_indices_within_bounding_box(pcd.points)
                    points_colors[indices] = np.array([1, 0, 0])
                    # vis.add_geometry(line_set)
                    for i in range(len(line_mesh_geoms)):
                        vis.add_geometry(line_mesh_geoms[i])
                    # o3d.visualization.draw_geometries(
                    #     [*line_mesh_geoms])
                pcd.colors = o3d.utility.Vector3dVector(points_colors)
                # num_pts = np.asarray(pcd.points).shape[0]
                # inds = np.random.choice(num_pts, int(num_pts/2), replace=False)
                # pcd = pcd.select_by_index(inds)


                vis.add_geometry(pcd)
                # o3d.visualization.draw_geometries([pcd, *line_set_list])
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

