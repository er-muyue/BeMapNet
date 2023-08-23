import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb as n_over_k
from scipy.optimize import linear_sum_assignment
from bemapnet.models.utils.mask_loss import SegmentationLoss
from bemapnet.models.utils.recovery_loss import PointRecoveryLoss
from bemapnet.utils.misc import nested_tensor_from_tensor_list
from bemapnet.utils.misc import get_world_size, is_available, is_distributed


class HungarianMatcher(nn.Module):

    def __init__(self, cost_obj=1., cost_ctr=1., cost_end=1., cost_mask=1., cost_curve=1., cost_recovery=1.,
                 ins_mask_loss_conf=None, point_loss_conf=None, class_weight=None):
        super().__init__()
        self.cost_obj, self.cost_ctr, self.cost_end = cost_obj, cost_ctr, cost_end
        self.cost_mask, self.cost_curve, self.cost_recovery = cost_mask, cost_curve, cost_recovery
        self.ins_mask_loss = SegmentationLoss(**ins_mask_loss_conf)
        self.recovery_loss = PointRecoveryLoss(**point_loss_conf)
        self.class_weight = class_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        num_decoders, num_classes = len(outputs["ins_masks"]), len(outputs["ins_masks"][0])
        matching_indices = [[[] for _ in range(num_classes)] for _ in range(num_decoders)]
        for dec_id in range(num_decoders):
            for cid in range(num_classes):
                if self.class_weight is not None and self.class_weight[cid] == 0:
                    continue

                bs, num_queries = outputs["obj_logits"][dec_id][cid].shape[:2]

                dt_probs = outputs["obj_logits"][dec_id][cid].flatten(0, 1).softmax(-1)  # [n_dt, 2], n_dt in a batch
                gt_idxes = torch.cat([tgt["obj_labels"][cid] for tgt in targets])        # [n_gt, ]
                cost_mat_obj = -dt_probs[:, gt_idxes]                                    # [n_dt, n_gt]

                dt_curves = outputs["curve_points"][dec_id][cid].flatten(0, 1)           # [n_dt, n, 2]
                dt_masks = outputs["ins_masks"][dec_id][cid].flatten(0, 1)               # [n_dt, h, w]
                gt_masks = torch.cat([tgt["ins_masks"][cid] for tgt in targets])         # [n_gt, h, w]
                cost_mat_mask, cost_mat_rec = 0, 0
                if gt_masks.shape[0] > 0:
                    dt_num, gt_num = dt_masks.shape[0], gt_masks.shape[0]
                    dt_masks = dt_masks.unsqueeze(1).expand(dt_num, gt_num, *dt_masks.shape[1:]).flatten(0, 1)
                    gt_masks = gt_masks.unsqueeze(0).expand(dt_num, gt_num, *gt_masks.shape[1:]).flatten(0, 1)
                    cost_mat_mask = self.ins_mask_loss(dt_masks, gt_masks, "matcher").reshape(dt_num, gt_num)
                    dt_curves = dt_curves.unsqueeze(1).expand(dt_num, gt_num, *dt_curves.shape[1:]).flatten(0, 1)
                    cost_mat_rec = self.recovery_loss(dt_curves, gt_masks).reshape(dt_num, gt_num)

                dt_ctrs = outputs["ctr_points"][dec_id][cid].flatten(0, 1).flatten(1)            # [n_dt, n, 2]
                gt_ctrs = torch.cat([tgt["ctr_points"][cid] for tgt in targets]).flatten(1)      # [n_gt, h, w]
                cost_mat_ctr = torch.cdist(dt_ctrs, gt_ctrs, p=1) / gt_ctrs.shape[1]

                dt_end_probs = outputs["end_logits"][dec_id][cid].flatten(0, 1).softmax(-1)
                gt_end_idxes = torch.cat([tgt["end_labels"][cid] for tgt in targets])
                cost_mat_end = -dt_end_probs[:, gt_end_idxes]                                    # [n_dt, n_gt]

                dt_curves = outputs["curve_points"][dec_id][cid].flatten(0, 1).flatten(1)        # [n_dt, n, 2]
                gt_curves = torch.cat([tgt["curve_points"][cid] for tgt in targets]).flatten(1)  # [n_gt, n, 2]
                cost_mat_curve = torch.cdist(dt_curves, gt_curves, p=1) / gt_curves.shape[1]

                sizes = [len(tgt["obj_labels"][cid]) for tgt in targets]
                C = self.cost_obj * cost_mat_obj + self.cost_mask * cost_mat_mask + \
                    self.cost_ctr * cost_mat_ctr + self.cost_end * cost_mat_end + self.cost_curve * cost_mat_curve +\
                    self.cost_recovery * cost_mat_rec
                C = C.view(bs, num_queries, -1).cpu()
                indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(C.split(sizes, -1))]
                matching_indices[dec_id][cid] = [self.to_tensor(i, j) for i, j in indices]

        return matching_indices

    @staticmethod
    def to_tensor(i, j):
        return torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)


class SetCriterion(nn.Module):

    def __init__(self, criterion_conf, matcher, num_degree, no_object_coe=1.0):
        super().__init__()
        self.num_degree = num_degree
        self.matcher = matcher
        self.criterion_conf = criterion_conf
        self.loss_weight_dict = self.criterion_conf['loss_weight']
        self.sem_mask_loss = SegmentationLoss(**criterion_conf['bev_decoder']['sem_mask_loss'])
        self.register_buffer("empty_weight", torch.tensor([1.0, no_object_coe]))

    def forward(self, outputs, targets):
        losses = {}
        matching_indices = self.matcher(outputs, targets)
        losses.update(self.criterion_instance(outputs, targets, matching_indices))
        losses.update(self.criterion_instance_labels(outputs, targets, matching_indices))
        losses.update(self.criterion_semantic_masks(outputs, targets))
        losses = {key: self.criterion_conf['loss_weight'][key] * losses[key] for key in losses}
        return sum(losses.values()), losses

    def criterion_instance(self, outputs, targets, matching_indices):
        loss_masks, loss_ctr, loss_end, loss_curve, loss_rec = 0, 0, 0, 0, 0
        device = outputs["ins_masks"][0][0].device
        num_decoders, num_classes = len(matching_indices), len(matching_indices[0])
        for i in range(num_decoders):
            w = self.criterion_conf['ins_decoder']['weight'][i]
            for j in range(num_classes):
                w2 = self.criterion_conf["class_weights"][j] if "class_weights" in self.criterion_conf else 1.0
                num_instances = sum(len(t["obj_labels"][j]) for t in targets)
                num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=device)
                if is_distributed() and is_available():
                    torch.distributed.all_reduce(num_instances)
                num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

                indices = matching_indices[i][j]
                src_idx = self._get_src_permutation_idx(indices)  # dt
                tgt_idx = self._get_tgt_permutation_idx(indices)  # gt

                # instance masks
                src_masks = outputs["ins_masks"][i][j][src_idx]
                tgt_masks = [t["ins_masks"][j] for t in targets]
                tgt_masks, _ = nested_tensor_from_tensor_list(tgt_masks).decompose()
                tgt_masks = tgt_masks.to(src_masks)[tgt_idx]
                loss_masks += w * self.matcher.ins_mask_loss(src_masks, tgt_masks, "loss").sum() / num_instances * w2

                # eof indices classification
                src_logits = outputs["end_logits"][i][j][src_idx]  # [M, K]
                tgt_labels = torch.cat([t["end_labels"][j][J] for t, (_, J) in zip(targets, indices)])  # (M, )
                loss_end += w * F.cross_entropy(src_logits, tgt_labels, ignore_index=-1, reduction='sum') / num_instances * w2

                # control points
                src_ctrs = outputs["ctr_points"][i][j][src_idx]  # [bs, num_queries, o, 2]
                end_labels = torch.max(src_logits.softmax(dim=-1), dim=-1)[1]  # [m, k]  0, 1, 2, 3...
                end_labels_new = (end_labels + 1) * self.num_degree[j] + 1
                valid_mask = torch.zeros(src_ctrs.shape[:2], device=device).long()
                for a, b in enumerate(end_labels_new):
                    valid_mask[a][:b] = 1
                src_ctrs_masked = (src_ctrs * valid_mask.unsqueeze(-1))
                tgt_ctrs = torch.zeros((len(tgt_idx[0]), *src_ctrs.shape[-2:]), device=device).float()
                valid_mask = torch.zeros((len(tgt_idx[0]), src_ctrs.shape[-2]), device=device).float()
                for idx in range(len(tgt_idx[0])):
                    batch_id, gt_id = tgt_idx[0][idx], tgt_idx[1][idx]
                    tgt_ctrs[idx] = targets[batch_id]['ctr_points'][j][gt_id]
                    valid_mask[idx] = targets[batch_id]['valid_masks'][j][gt_id]
                tgt_ctrs_masked = (tgt_ctrs * valid_mask.unsqueeze(-1))
                num_pt = src_ctrs.shape[-2] * src_ctrs.shape[-1]
                loss_ctr += w * F.l1_loss(src_ctrs_masked, tgt_ctrs_masked, reduction='sum') / num_instances / num_pt * w2

                # curve loss
                src_curves = outputs["curve_points"][i][j][src_idx]
                tgt_curves = torch.zeros((len(tgt_idx[0]), *src_curves.shape[-2:]), device=device).float()
                for idx in range(len(tgt_idx[0])):
                    batch_id, gt_id = tgt_idx[0][idx], tgt_idx[1][idx]
                    tgt_curves[idx] = targets[batch_id]['curve_points'][j][gt_id]
                num_pt = src_curves.shape[-2] * src_curves.shape[-1]
                loss_curve += w * F.l1_loss(src_curves, tgt_curves, reduction='sum') / num_instances / num_pt * w2

                # recovery loss
                loss_rec += w * self.matcher.recovery_loss(src_curves, tgt_masks).sum() / num_instances * w2

        loss_masks /= (num_decoders * num_classes)
        loss_ctr /= (num_decoders * num_classes)
        loss_curve /= (num_decoders * num_classes)
        loss_end /= (num_decoders * num_classes)
        loss_rec /= (num_decoders * num_classes)

        return {"ctr_loss": loss_ctr, "end_loss": loss_end, "msk_loss": loss_masks, "curve_loss": loss_curve,
                "recovery_loss": loss_rec}

    def criterion_instance_labels(self, outputs, targets, matching_indices):
        loss_labels = 0
        num_decoders, num_classes = len(matching_indices), len(matching_indices[0])
        for i in range(num_decoders):
            w = self.criterion_conf['ins_decoder']['weight'][i]
            for j in range(num_classes):
                w2 = self.criterion_conf["class_weights"][j] if "class_weights" in self.criterion_conf else 1.0
                indices = matching_indices[i][j]
                idx = self._get_src_permutation_idx(indices)  # (batch_id, query_id)
                logits = outputs["obj_logits"][i][j]
                target_classes_o = torch.cat([t["obj_labels"][j][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(logits.shape[:2], 1, dtype=torch.int64, device=logits.device)
                target_classes[idx] = target_classes_o
                loss_labels += (w * F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)) * w2
        loss_labels /= (num_decoders * num_classes)
        return {"obj_loss": loss_labels}

    def criterion_semantic_masks(self, outputs, targets):
        loss_masks = 0
        num_decoders, num_classes = len(outputs["sem_masks"]), len(outputs["sem_masks"][0])
        for i in range(num_decoders):
            w = self.criterion_conf['bev_decoder']['weight'][i]
            for j in range(num_classes):
                w2 = self.criterion_conf["class_weights"][j] if "class_weights" in self.criterion_conf else 1.0
                dt_masks = outputs["sem_masks"][i][j]  # (B, 2, H, W)
                gt_masks = torch.stack([t["sem_masks"][j] for t in targets], dim=0)  # (B, H, W)
                loss_masks += w * self.sem_mask_loss(dt_masks[:, 1, :, :], gt_masks, "loss").mean() * w2
        loss_masks /= num_decoders
        return {"sem_loss": loss_masks}

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class PiecewiseBezierMapPostProcessor(nn.Module):
    def __init__(self, criterion_conf, matcher_conf, bezier_conf, map_conf, no_object_coe=1.0):
        super(PiecewiseBezierMapPostProcessor, self).__init__()
        # setting
        self.num_classes = map_conf['num_classes']
        self.ego_size = map_conf['ego_size']
        self.map_size = map_conf['map_size']
        self.line_width = map_conf['line_width']
        self.num_degree = bezier_conf['num_degree']
        self.num_pieces = bezier_conf['max_pieces']
        self.num_points = bezier_conf['num_points']
        self.curve_size = bezier_conf['piece_length']
        self.class_indices = torch.tensor(list(range(self.num_classes)), dtype=torch.int).cuda()
        self.bezier_coefficient_np = self._get_bezier_coefficients()
        self.bezier_coefficient = [torch.from_numpy(x).float().cuda() for x in self.bezier_coefficient_np]
        self.matcher = HungarianMatcher(**matcher_conf)
        self.criterion = SetCriterion(criterion_conf, self.matcher, self.num_degree, no_object_coe)
        self.save_thickness = map_conf['save_thickness'] if 'save_thickness' in map_conf else 1

    def forward(self, outputs, targets=None):
        outputs.update(self.bezier_curve_outputs(outputs))
        if self.training:
            targets = self.refactor_targets(targets)
            return self.criterion.forward(outputs, targets)
        else:
            return self.post_processing(outputs)

    def bezier_curve_outputs(self, outputs):
        dt_ctr_im, dt_ctr_ex, dt_ends = outputs["ctr_im"], outputs["ctr_ex"], outputs["end_logits"]
        num_decoders, num_classes = len(dt_ends), len(dt_ends[0])
        ctr_points = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        curve_points = [[[] for _ in range(self.num_classes)] for _ in range(num_decoders)]
        for i in range(num_decoders):
            for j in range(num_classes):
                batch_size, num_queries = dt_ctr_im[i][j].shape[:2]

                im_coords = dt_ctr_im[i][j].sigmoid()
                ex_offsets = dt_ctr_ex[i][j].sigmoid() - 0.5
                im_center_coords = ((im_coords[:, :, :-1] + im_coords[:, :, 1:]) / 2).unsqueeze(-2)
                ex_coords = torch.stack([im_center_coords[:, :, :, :, 0] + ex_offsets[:, :, :, :, 0],
                                         im_center_coords[:, :, :, :, 1] + ex_offsets[:, :, :, :, 1]], dim=-1)
                im_coords = im_coords.unsqueeze(-2)
                ctr_coords = torch.cat([im_coords[:, :, :-1], ex_coords], dim=-2).flatten(2, 3)
                ctr_coords = torch.cat([ctr_coords, im_coords[:, :, -1:, 0, :]], dim=-2)
                ctr_points[i][j] = ctr_coords.clone()

                end_inds = torch.max(torch.softmax(dt_ends[i][j].flatten(0, 1), dim=-1), dim=-1)[1]
                curve_pts = self.curve_recovery_with_bezier(ctr_coords.flatten(0, 1), end_inds, j)
                curve_points[i][j] = curve_pts.reshape(batch_size, num_queries, *curve_pts.shape[-2:])

        return {"curve_points": curve_points, 'ctr_points': ctr_points}

    def refactor_targets(self, targets):
        targets_refactored = []
        batch_size, num_classes = len(targets["masks"]), len(targets["masks"][0])
        targets["masks"] = targets["masks"].cuda()
        targets["points"] = targets["points"].cuda()
        targets["labels"] = targets["labels"].cuda()
        for batch_id in range(batch_size):

            sem_masks, ins_masks, ins_objects = [], [], []
            ctr_points, curve_points, end_labels, valid_masks = [], [], [], []

            ins_classes = targets['labels'][batch_id][:, 0].int()
            cls_ids, ins_ids = torch.where((ins_classes.unsqueeze(0) == self.class_indices.unsqueeze(1)).int())
            for cid in range(num_classes):
                indices = ins_ids[torch.where(cls_ids == cid)]
                num_ins = indices.shape[0]

                # object class: 0 or 1
                ins_obj = torch.zeros((num_ins,), dtype=torch.long).cuda()
                ins_objects.append(ins_obj)

                # bezier control points coords
                num_max = self.num_points[cid]
                ctr_pts = targets['points'][batch_id][indices][:, :num_max].float()
                ctr_pts[:, :, 0] = ctr_pts[:, :, 0] / self.ego_size[1]
                ctr_pts[:, :, 1] = ctr_pts[:, :, 1] / self.ego_size[0]
                ctr_points.append(ctr_pts)

                # piecewise end indices
                end_indices = targets['labels'][batch_id][indices][:, 1].long()
                end_labels.append(end_indices)

                # bezier valid masks
                v_mask = torch.zeros((num_ins, num_max), dtype=torch.int8).cuda()
                for ins_id in range(num_ins):
                    k = targets['labels'][batch_id][indices[ins_id]][2].long()
                    v_mask[ins_id][:k] = 1
                valid_masks.append(v_mask)

                # curve points
                curve_pts = self.curve_recovery_with_bezier(ctr_pts, end_indices, cid)
                curve_points.append(curve_pts)

                # instance mask
                mask_pc = targets["masks"][batch_id][cid]  # mask supervision
                unique_ids = torch.unique(mask_pc, sorted=True)[1:]
                if num_ins == unique_ids.shape[0]:
                    ins_msk = (mask_pc.unsqueeze(0).repeat(num_ins, 1, 1) == unique_ids.view(-1, 1, 1)).float()
                else:
                    ins_msk = np.zeros((num_ins, *self.map_size), dtype=np.uint8)
                    for i, ins_pts in enumerate(curve_pts):
                        ins_pts[:, 0] *= self.map_size[1]
                        ins_pts[:, 1] *= self.map_size[0]
                        ins_pts = ins_pts.cpu().data.numpy().astype(np.int32)
                        cv2.polylines(ins_msk[i], [ins_pts], False, color=1, thickness=self.line_width)
                    ins_msk = torch.from_numpy(ins_msk).float().cuda()
                ins_masks.append(ins_msk)

                # semantic mask
                sem_msk = (ins_msk.sum(0) > 0).float()
                sem_masks.append(sem_msk)

            targets_refactored.append({
                "sem_masks": sem_masks, "ins_masks": ins_masks, "obj_labels": ins_objects,
                "ctr_points": ctr_points, "end_labels": end_labels, "curve_points": curve_points,
                "valid_masks": valid_masks,
            })

        return targets_refactored

    def curve_recovery_with_bezier(self, ctr_points, end_indices, cid):
        device = ctr_points.device
        curve_pts_ret = torch.zeros((0, self.curve_size, 2), dtype=torch.float, device=device)
        num_instances, num_pieces = ctr_points.shape[0], ctr_points.shape[1]
        pieces_ids = [[i+j for j in range(self.num_degree[cid]+1)] for i in range(0, num_pieces - 1, self.num_degree[cid])]
        pieces_ids = torch.tensor(pieces_ids).long().to(device)
        points_ids = torch.tensor(list(range(self.curve_size))).long().to(device)
        points_ids = (end_indices + 1).unsqueeze(1) * points_ids.unsqueeze(0)
        if num_instances > 0:
            ctr_points_flatten = ctr_points[:, pieces_ids, :].flatten(0, 1)
            curve_pts = torch.matmul(self.bezier_coefficient[cid], ctr_points_flatten)
            curve_pts = curve_pts.reshape(num_instances, pieces_ids.shape[0], *curve_pts.shape[-2:])
            curve_pts = curve_pts.flatten(1, 2)
            curve_pts_ret = torch.stack([curve_pts[i][points_ids[i]] for i in range(points_ids.shape[0])])
        return curve_pts_ret

    def _get_bezier_coefficients(self):

        def bernstein_func(n, t, k):
            return (t ** k) * ((1 - t) ** (n - k)) * n_over_k(n, k)

        ts = np.linspace(0, 1, self.curve_size)
        bezier_coefficient_list = []
        for nn in self.num_degree:
            bezier_coefficient_list.append(np.array([[bernstein_func(nn, t, k) for k in range(nn + 1)] for t in ts]))
        return bezier_coefficient_list

    def post_processing(self, outputs):
        batch_results, batch_masks, batch_masks5 = [], [], []
        batch_size = outputs["obj_logits"][-1][0].shape[0]
        for i in range(batch_size):
            points, scores, labels = [None], [-1], [0]
            masks = np.zeros((self.num_classes, *self.map_size)).astype(np.uint8)
            masks5 = np.zeros((self.num_classes, *self.map_size)).astype(np.uint8)
            instance_index = 1
            for j in range(self.num_classes):
                pred_scores, pred_labels = torch.max(F.softmax(outputs["obj_logits"][-1][j][i], dim=-1), dim=-1)
                keep_ids = torch.where((pred_labels == 0).int())[0]
                if keep_ids.shape[0] == 0:
                    continue
                curve_pts = outputs['curve_points'][-1][j][i][keep_ids].cpu().data.numpy()
                curve_pts[:, :, 0] *= self.map_size[1]
                curve_pts[:, :, 1] *= self.map_size[0]
                for dt_curve, dt_score in zip(curve_pts, pred_scores[keep_ids]):
                    cv2.polylines(masks[j], [dt_curve.astype(np.int32)], False, color=instance_index,
                                  thickness=self.save_thickness)
                    cv2.polylines(masks5[j], [dt_curve.astype(np.int32)], False, color=instance_index, thickness=12)
                    instance_index += 1
                    points.append(curve_pts)
                    scores.append(self._to_np(dt_score).item())
                    labels.append(j + 1)
            batch_results.append({'map': points, 'confidence_level': scores, 'pred_label': labels})
            batch_masks.append(masks)
            batch_masks5.append(masks5)
        return batch_results, batch_masks, batch_masks5

    @staticmethod
    def _to_np(tensor):
        return tensor.cpu().data.numpy()
