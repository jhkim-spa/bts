import numpy as np
import plotly.graph_objs as go
import open3d as o3d
from open3d import geometry


def depth_to_pc(depth):
    fx, fy = 388, 388
    cx, cy = 322.0336, 236.3574

    z = depth.astype(float) / 1000.0  

    px, py = np.meshgrid(np.arange(640), np.arange(480))  # pixel_x, pixel_y
    px, py = px.astype(float), py.astype(float)
    x = ((px - cx) / fx) * z 
    y = ((py - cy) / fy) * z 

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    return (x, y, z)

def scatter_3d(x, y, z):
    data = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=1,
        colorscale='Jet',
        line=dict(
            width=0.0
        ),
        opacity=0.8
        )
    )
    fig = go.Figure(data=[data])
    fig.show()

def visualize(pcd, line_set_list, lookat, front, up, zoom, save_path):
    vis = o3d.visualization.Visualizer()
    for i in range(len(line_set_list)):
        vis.add_geometry(line_set_list[i])
    view_ctl = vis.get_view_control()
    view_ctl.set_lookat(lookat)
    view_ctl.set_front(front)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)
    vis.run()
    vis.capture_screen_image(save_path, True)
    vis.clear_geometries()
    vis.destroy_window()

import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


# def main():
#     print("Demonstrating LineMesh vs LineSet")
#     # Create Line Set
#     points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
#               [0, 1, 1], [1, 1, 1]]
#     lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
#              [0, 4], [1, 5], [2, 6], [3, 7]]
#     colors = [[1, 0, 0] for i in range(len(lines))]

#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(colors)

#     # Create Line Mesh 1
#     points = np.array(points) + [0, 0, 2]
#     line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
#     line_mesh1_geoms = line_mesh1.cylinder_segments

#     # Create Line Mesh 1
#     points = np.array(points) + [0, 2, 0]
#     line_mesh2 = LineMesh(points, radius=0.03)
#     line_mesh2_geoms = line_mesh2.cylinder_segments

#     o3d.visualization.draw_geometries(
#         [line_set, *line_mesh1_geoms, *line_mesh2_geoms])