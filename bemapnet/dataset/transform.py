import cv2
import mmcv
import torch
import numpy as np
from PIL import Image
from collections.abc import Sequence


class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, data_dict):
        imgs = []
        for img in data_dict["images"]:
            if self.to_rgb:
                img = img.astype(np.float32) / 255.0
            img = self.im_normalize(img, self.mean, self.std, self.to_rgb)
            imgs.append(img)
        data_dict["images"] = imgs
        data_dict["extra_infos"]["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return data_dict

    @staticmethod
    def im_normalize(img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8  # cv2 inplace normalization does not accept uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img


class ToTensor(object):
    """Default formatting bundle."""

    def __call__(self, data_dict):
        """Call function to transform and format common fields in data_dict.

        Args:
            data_dict (dict): Data dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with default bundle.
        """

        for k in ["images", "extrinsic", "intrinsic", "ida_mats"]:
            if k == "images":
                data_dict[k] = np.stack([img.transpose(2, 0, 1) for img in data_dict[k]], axis=0)
            data_dict[k] = self.to_tensor(np.ascontiguousarray(data_dict[k]))

        for k in ["masks", "points", "labels"]:
            data_dict["targets"][k] = self.to_tensor(np.ascontiguousarray(data_dict["targets"][k]))

        return data_dict

    @staticmethod
    def to_tensor(data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not mmcv.is_str(data):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")
