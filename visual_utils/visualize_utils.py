"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def check_scene(points, bbox, min_points):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    _, area_bbox = translate_boxes_to_open3d_instance(bbox)
    pts = pts.crop(area_bbox)
    return len(np.asarray(pts.points)) > min_points



def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
                point_colors=None, draw_origin=True, area=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.5
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if area is not None:
        _, area_bbox = translate_boxes_to_open3d_instance(area)
        pts = pts.crop(area_bbox)
        pts = pts.translate(-1 * area[:3])
        translation_vector = -1 * area[:3]
        max_bound = pts.get_max_bound()
        min_bound = pts.get_min_bound()
    else:
        max_bound = None
        min_bound = None
        translation_vector = None

    if gt_boxes is not None:
        vis, _ = draw_box(vis, gt_boxes, (0, 0, 1),
                          min_bound=min_bound, max_bound=max_bound, translation=translation_vector)

    if ref_boxes is not None:
        vis, colored_box = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, score=ref_scores,
                                    min_bound=min_bound, max_bound=max_bound, translation=translation_vector)
        # Paint points in bbox
        indices = colored_box.get_point_indices_within_bounding_box(pts.points)
        colored_points = pts.select_by_index(indices, invert=False)
        shape = (len(np.asarray(colored_points.points)), 1)
        colors = np.hstack((np.ones(shape), np.zeros(shape), np.zeros(shape)))
        colored_points.colors = open3d.utility.Vector3dVector(colors)

        other_points = pts.select_by_index(indices, invert=True)
        vis.add_geometry(colored_points)
        vis.add_geometry(other_points)
    else:
        vis.add_geometry(pts)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, min_bound=None, max_bound=None,
             translation=None, eps=0.5):
    colored_box = None
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        if translation is not None:
            box3d = box3d.translate(translation)
            line_set = line_set.translate(translation)

        if min_bound is not None and max_bound is not None:
            if np.all(box3d.get_max_bound()[:2] < max_bound[:2] + eps) and \
                    np.all(box3d.get_min_bound()[:2] > min_bound[:2] - 0.5):
                if score is not None:
                    print(f"Current IOU estimates score {score.item()}")
                    colored_box = box3d
                else:
                    print(f"There is intersection with gt")
                vis.add_geometry(line_set)
    return vis, colored_box
