import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness


class SegmentationLoss(nn.Module):

    def __init__(self, ce_weight, dice_weight, use_point_render, num_points=8000, oversample=3.0, importance=0.75):
        super(SegmentationLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.use_point_render = use_point_render
        self.num_points = num_points
        self.oversample = oversample
        self.importance = importance

    def forward(self, dt_masks, gt_masks, stage="loss"):
        loss = 0
        if self.use_point_render:
            dt_masks, gt_masks = self.points_render(dt_masks, gt_masks, stage)
        if self.ce_weight > 0:
            loss += self.ce_weight * self.forward_sigmoid_ce_loss(dt_masks, gt_masks)
        if self.dice_weight > 0:
            loss += self.dice_weight * self.forward_dice_loss(dt_masks, gt_masks)
        return loss

    @staticmethod
    def forward_dice_loss(inputs, targets):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    @staticmethod
    def forward_sigmoid_ce_loss(inputs, targets):
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        return loss.mean(1)

    def points_render(self, src_masks, tgt_masks, stage):
        assert stage in ["loss", "matcher"]
        assert src_masks.shape == tgt_masks.shape

        src_masks = src_masks[:, None]
        tgt_masks = tgt_masks[:, None]

        if stage == "matcher":
            point_coords = torch.rand(1, self.num_points, 2, device=src_masks.device)
            point_coords_src = point_coords.repeat(src_masks.shape[0], 1, 1)
            point_coords_tgt = point_coords.repeat(tgt_masks.shape[0], 1, 1)
        else:
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample,
                self.importance,
            )
            point_coords_src = point_coords.clone()
            point_coords_tgt = point_coords.clone()

        src_masks = point_sample(src_masks, point_coords_src, align_corners=False).squeeze(1)
        tgt_masks = point_sample(tgt_masks, point_coords_tgt, align_corners=False).squeeze(1)

        return src_masks, tgt_masks

    @staticmethod
    def calculate_uncertainty(logits):
        """
        We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
            foreground class in `classes`.
        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                class-agnostic, where R is the total number of predicted masks in all images and C is
                the number of foreground classes. The values are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                the most uncertain locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))
