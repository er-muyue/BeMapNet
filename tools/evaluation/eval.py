import os
import sys
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from ap import instance_mask_ap as get_batch_ap


class BeMapNetResultForNuScenes(Dataset):
    def __init__(self, gt_dir, dt_dir, val_txt):
        self.gt_dir, self.dt_dir = gt_dir, dt_dir
        self.tokens = [fname.strip().split('.')[0] for fname in open(val_txt).readlines()]        
        self.max_line_count = 100

    def __getitem__(self, idx):
        token = self.tokens[idx]
        gt_path = os.path.join(self.gt_dir, f"{token}.npz")
        gt_masks = np.load(open(gt_path, "rb"), allow_pickle=True)["instance_mask"]
        dt_item = np.load(os.path.join(self.dt_dir, f"{token}.npz"), allow_pickle=True)
        dt_masks = dt_item["dt_mask"]
        dt_scores = dt_item['dt_res'].item()["confidence_level"]
        dt_scores = np.array(list(dt_scores) + [-1] * (self.max_line_count - len(dt_scores)))
        return torch.from_numpy(dt_masks), torch.from_numpy(dt_scores).float(), torch.from_numpy(gt_masks)

    def __len__(self):
        return len(self.tokens)


class BeMapNetEvaluatorForNuScenes(object):
    def __init__(self, gt_dir, dt_dir, val_txt, batch_size=4, num_classes=3, map_resolution=(0.15, 0.15)):

        self.THRESHOLDS = [0.2, 0.5, 1.0, 1.5]
        self.CLASS_NAMES = ["Divider", "PedCross", "Contour"]
        self.SAMPLED_RECALLS = torch.linspace(0.1, 1, 10).cuda()
        self.res_dataloader = DataLoader(
            BeMapNetResultForNuScenes(gt_dir, dt_dir, val_txt), 
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8
        )
        self.map_resolution = map_resolution
        self.ap_matrix = torch.zeros((num_classes, len(self.THRESHOLDS))).cuda()
        self.ap_count_matrix = torch.zeros((num_classes, len(self.THRESHOLDS))).cuda()

    def execute(self):

        for dt_masks, dt_scores, gt_masks in tqdm(self.res_dataloader):
            self.ap_matrix, self.ap_count_matrix = get_batch_ap(
                self.ap_matrix,
                self.ap_count_matrix,
                dt_masks.cuda(),
                gt_masks.cuda(),
                *self.map_resolution,
                dt_scores.cuda(),
                self.THRESHOLDS,
                self.SAMPLED_RECALLS,
            )
        ap = (self.ap_matrix / self.ap_count_matrix).cpu().data.numpy()
        self._format_print(ap)

    def _format_print(self, ap):
        res_matrix = []
        table_header = ["Class", "AP@.2", "AP@.5", "AP@1.", "AP@1.5", "mAP@HARD", "mAP@EASY"]
        table_values = []
        for i, cls_name in enumerate(self.CLASS_NAMES):
            res_matrix_line = [ap[i][0], ap[i][1], ap[i][2], ap[i][3], np.mean(ap[i][:-1]), np.mean(ap[i][1:])]
            res_matrix.append(res_matrix_line)
            table_values.append([cls_name] + self.line_data_to_str(*res_matrix_line))
        avg = np.mean(np.array(res_matrix), axis=0)
        table_values.append(["Average", *self.line_data_to_str(*avg)])
        table_str = tabulate(table_values, headers=table_header, tablefmt="grid")
        print(table_str)
        return table_str

    @staticmethod
    def line_data_to_str(ap0, ap1, ap2, ap3, map1, map2):
        return [
            "{:.1f}".format(ap0 * 100),
            "{:.1f}".format(ap1 * 100),
            "{:.1f}".format(ap2 * 100),
            "{:.1f}".format(ap3 * 100),
            "{:.1f}".format(map1 * 100),
            "{:.1f}".format(map2 * 100),
        ]


evaluator = BeMapNetEvaluatorForNuScenes(
    gt_dir=sys.argv[1],
    dt_dir=sys.argv[2],
    val_txt=sys.argv[3],
    batch_size=4,
    num_classes=3,
    map_resolution=(0.15, 0.15),
)

evaluator.execute()
