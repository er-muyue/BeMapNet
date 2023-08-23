import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, box
from tools.bezier_converter.bezier import PiecewiseBezierCurve


class RasterizedLocalMap(object):
    def __init__(self, patch_size, canvas_size, num_degrees, max_channel, thickness, patch_angle=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.max_channel = max_channel
        self.num_degrees = num_degrees
        self.thickness = thickness
        assert self.thickness[0] == 1
        self.patch_box = (0.0, 0.0, self.patch_size[0], self.patch_size[1])
        self.patch_angle = patch_angle
        self.patch = self.get_patch_coord()
        self.pbc_funcs = {
            d: PiecewiseBezierCurve(num_points=100, num_degree=d, margin=0.05, threshold=0.1) for d in num_degrees
        }

    def convert_vec_to_mask(self, vectors):
        vector_num_list = {cls_idx: [] for cls_idx in range(self.max_channel)}  # map-type -> list
        for vector in vectors:
            if vector['pts_num'] >= 2:
                vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))
        ins_idx = 1  # instance-index
        instance_masks = np.zeros(
            (len(self.thickness), self.max_channel, self.canvas_size[1], self.canvas_size[0]), np.uint8)
        instance_vec_points, instance_ctr_points = [], []
        for cls_idx in range(self.max_channel):
            pbc_func = self.pbc_funcs[self.num_degrees[cls_idx]]
            masks, map_points, ctr_points, ins_idx = self.line_geom_to_mask(vector_num_list[cls_idx], ins_idx, pbc_func)
            instance_masks[:, cls_idx, :, :] = masks
            for pts in map_points:
                instance_vec_points.append({'pts': pts, 'pts_num': len(pts), 'type': cls_idx})
            for pts in ctr_points:
                instance_ctr_points.append({'pts': pts, 'pts_num': len(pts), 'type': cls_idx})
        instance_masks = np.stack(instance_masks).astype(np.uint8)
        semantic_masks = (instance_masks != 0).astype(np.uint8)
        return semantic_masks, instance_masks, instance_vec_points, instance_ctr_points

    def line_geom_to_mask(self, layer_geom, idx, pbc_func, trans_type='index'):
        patch_x, patch_y, patch_h, patch_w = self.patch_box
        canvas_h = self.canvas_size[0]
        canvas_w = self.canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0
        map_masks = np.zeros((len(self.thickness), *self.canvas_size), np.uint8)
        map_points, ctr_points = [], []
        for line in layer_geom:
            new_line = line.intersection(self.patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                if new_line.geom_type == 'MultiLineString':
                    for single_line in new_line:
                        pts2 = self.patch_size - np.array(single_line.coords[:])[:, ::-1]
                        ctr_points.append(pbc_func(pts2))
                        single_line = affinity.scale(single_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
                        map_masks, idx = self.mask_for_lines(single_line, map_masks, self.thickness, idx, trans_type)
                        pts = self.canvas_size - np.array(single_line.coords[:])[:, ::-1]
                        map_points.append(pts.tolist())
                else:
                    pts2 = self.patch_size - np.array(new_line.coords[:])[:, ::-1]
                    ctr_points.append(pbc_func(pts2))
                    new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
                    map_masks, idx = self.mask_for_lines(new_line, map_masks, self.thickness, idx, trans_type)
                    pts = self.canvas_size - np.array(new_line.coords[:])[:, ::-1]
                    map_points.append(pts.tolist())
        map_masks_ret = []
        for i in range(len(self.thickness)):
            map_masks_ret.append(np.flip(np.rot90(map_masks[i][None], k=1, axes=(1, 2)), axis=2)[0])
        map_masks_ret = np.array(map_masks_ret)
        return map_masks_ret, map_points, ctr_points, idx

    @staticmethod
    def mask_for_lines(lines, mask, thickness, idx, trans_type='index'):
        coords = np.asarray(list(lines.coords), np.int32)
        coords = coords.reshape((-1, 2))
        if len(coords) < 2:
            return mask, idx
        for i, t in enumerate(thickness):
            if trans_type == 'index':
                cv2.polylines(mask[i], [coords], False, color=idx, thickness=t)
                idx += 1
        return mask, idx

    def get_patch_coord(self):
        patch_x, patch_y, patch_h, patch_w = self.patch_box
        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0
        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, self.patch_angle, origin=(patch_x, patch_y), use_radians=False)
        return patch
