import json
import pickle
from copy import deepcopy
from glob import glob
import hydra
import numpy as np
from omegaconf import DictConfig
from visual_utils import visualize_utils as V
import torch


def load_data(pc_folder: str, result_path: str, gt_folder: str):
    scenes = sorted(glob(f"{pc_folder}/*.pkl"), key=lambda x: int(x.split('_')[-1][:-4]))
    gt_files = sorted(glob(f"{gt_folder}/*.pkl"), key=lambda x: int(x.split('_')[-1][:-4]))
    results = json.load(open(result_path))
    points = list()
    gts = list()

    for scene, gt_file in zip(scenes, gt_files):
        points.append(pickle.load(open(scene, "rb"))['points'][:, 1:])
        gts.append(pickle.load(open(gt_file, "rb"))[0])
    return list(zip(points, results, gts))


def extend(box, scale=2):
    extended_box = deepcopy(box)
    extended_box[3:6] = scale * box[3:6]
    return extended_box


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data = load_data(cfg['Data']['pc_folder'], cfg['Data']['results_path'], cfg['Data']['gt_folder'])
    for (points, results, gt) in data:
        # Set usefull outputs
        useful_keys = ('pred_iou_scores', 'boxes_lidar', 'score', 'name')
        # Filter results by predicted IoU
        scores = torch.tensor(results['pred_iou_scores'])
        indexes = np.array(torch.sigmoid(scores)) > cfg['Params']['iou threshold']
        indexes = np.squeeze(indexes)
        filtered_results = dict()
        for key in useful_keys:
            filtered_results[key] = np.array(results[key])[indexes]
        # Crop area around prediction
        for box, score in zip(filtered_results['boxes_lidar'], filtered_results['pred_iou_scores']):
            if np.any(box[3:6] < np.array(cfg['Params']['size th'])):
                continue
            extended_box = extend(box, scale=3)
            print("===========================")
            if V.check_scene(points=points, bbox=box, min_points=cfg['Params']['min points']):
                V.draw_scenes(
                    points=points, ref_boxes=np.array([box]),
                    ref_scores=torch.sigmoid(torch.tensor([score])), ref_labels=None,
                    gt_boxes=gt, area=extended_box
                )


if __name__ == '__main__':
    main()