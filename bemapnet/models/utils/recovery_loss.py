import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample


class PointRecoveryLoss(nn.Module):

    def __init__(self, ce_weight, dice_weight, curve_width, tgt_shape):
        super(PointRecoveryLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.kernel = self.generate_kernel(curve_width, tgt_shape)

    def forward(self, points, gt_masks):
        points_expanded = points.unsqueeze(2) - self.kernel.repeat(points.shape[0], 1, 1, 1)
        points_expanded = torch.clamp(points_expanded.flatten(1, 2), min=0, max=1)  # (N, P*w*w, 2) [0, 1]
        dt_points = point_sample(gt_masks[:, None], points_expanded, align_corners=False).squeeze(1).flatten(1)
        gt_points = torch.ones_like(dt_points)
        loss = 0
        if self.ce_weight > 0:
            loss += self.ce_weight * self.forward_ce_loss(dt_points, gt_points)
        if self.dice_weight > 0:
            loss += self.dice_weight * self.forward_dice_loss(dt_points, gt_points)
        return loss

    @staticmethod
    def generate_kernel(curve_width, tgt_shape, device='cuda'):
        width = torch.tensor(list(range(curve_width)))
        kernel = torch.stack(torch.meshgrid(width, width), dim=-1).float()
        kernel = kernel - curve_width // 2
        kernel[..., 0] = kernel[..., 0] / tgt_shape[1]
        kernel[..., 1] = kernel[..., 1] / tgt_shape[0]
        kernel = kernel.flatten(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, w*w, 2)
        kernel = kernel.cuda() if device == 'cuda' else kernel
        return kernel

    @staticmethod
    def forward_dice_loss(inputs, targets):
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    @staticmethod
    def forward_ce_loss(inputs, targets):
        loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        return loss.mean(1)
