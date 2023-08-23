import torch
import numpy as np
from shapely.geometry import LineString
from scipy.special import comb as n_over_k


class PiecewiseBezierCurve(object):
    def __init__(self, num_points=100, num_degree=2, margin=0.05, threshold=0.1):
        super().__init__()
        self.num_points = num_points
        self.num_degree = num_degree
        self.margin = margin
        self.bezier_coefficient = self._get_bezier_coefficients(np.linspace(0, 1, self.num_points))
        self.threshold = threshold

    def _get_bezier_coefficients(self, t_list):
        bernstein_fn = lambda n, t, k: (t ** k) * ((1 - t) ** (n - k)) * n_over_k(n, k)
        bezier_coefficient_fn = \
            lambda ts: [[bernstein_fn(self.num_degree, t, k) for k in range(self.num_degree + 1)] for t in t_list]
        return np.array(bezier_coefficient_fn(t_list))

    def _get_interpolated_points(self, points):
        line = LineString(points)
        distances = np.linspace(0, line.length, self.num_points)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        return sampled_points

    def _get_chamfer_distance(self, points_before, points_after):
        points_before = torch.from_numpy(points_before).float()
        points_after = torch.from_numpy(points_after).float()
        dist = torch.cdist(points_before, points_after)
        dist1, _ = torch.min(dist, 2)
        dist1 = (dist1 * (dist1 > self.margin).float())
        dist2, _ = torch.min(dist, 1)
        dist2 = (dist2 * (dist2 > self.margin).float())
        return (dist1.mean(-1) + dist2.mean(-1)) / 2

    def bezier_fitting(self, curve_pts):
        curve_pts_intered = self._get_interpolated_points(curve_pts)
        bezier_ctrl_pts = np.linalg.pinv(self.bezier_coefficient).dot(curve_pts_intered)
        bezier_ctrl_pts = np.concatenate([curve_pts[0:1], bezier_ctrl_pts[1:-1], curve_pts[-1:]], axis=0)
        curve_pts_recovery = self.bezier_coefficient.dot(bezier_ctrl_pts)
        criterion = self._get_chamfer_distance(curve_pts_intered[None, :, :], curve_pts_recovery[None, :, :]).item()
        return bezier_ctrl_pts, criterion

    @staticmethod
    def sequence_reverse(ctr_points):
        ctr_points = np.array(ctr_points)
        (xs, ys), (xe, ye) = ctr_points[0], ctr_points[-1]
        if ys > ye:
            ctr_points = ctr_points[::-1]
        return ctr_points

    def __call__(self, curve_pts):
        ctr_points_piecewise = []
        num_points = curve_pts.shape[0]
        start, end = 0, num_points - 1
        while start < end:
            ctr_points, loss = self.bezier_fitting(curve_pts[start: end + 1])
            if loss < self.threshold:
                start, end = end, num_points - 1
                if start >= end:
                    ctr_points_piecewise += ctr_points.tolist()
                else:
                    ctr_points_piecewise += ctr_points.tolist()[:-1]
            else:
                end = end - 1
        ctr_points_piecewise = self.sequence_reverse(ctr_points_piecewise)
        return ctr_points_piecewise
