import os
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from skimage import io as skimage_io
from torch.utils.data import Dataset


class NuScenesMapDataset(Dataset):
    def __init__(self, img_key_list, map_conf, ida_conf, bezier_conf, transforms, data_split="training"):
        super().__init__()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        self.ida_conf = ida_conf
        self.bez_conf = bezier_conf
        self.ego_size = map_conf["ego_size"]
        self.mask_key = map_conf["mask_key"]
        self.nusc_root = map_conf["nusc_root"]
        self.anno_root = map_conf["anno_root"]
        self.split_dir = map_conf["split_dir"]
        self.num_degree = bezier_conf["num_degree"]
        self.max_pieces = bezier_conf["max_pieces"]
        self.max_instances = bezier_conf["max_instances"]
        self.split_mode = 'train' if data_split == "training" else 'val'
        split_path = os.path.join(self.split_dir, f'{self.split_mode}.txt')
        self.tokens = [token.strip() for token in open(split_path).readlines()]
        self.transforms = transforms

    def __getitem__(self, idx: int):
        token = self.tokens[idx]
        sample = np.load(os.path.join(self.anno_root, f'{token}.npz'), allow_pickle=True)
        resize_dims, crop, flip, rotate = self.sample_ida_augmentation()
        images, ida_mats = [], []
        for im_view in self.img_key_list:
            for im_path in sample['image_paths']:
                if im_path.startswith(f'samples/{im_view}/'):
                    im_path = os.path.join(self.nusc_root, im_path)
                    img = skimage_io.imread(im_path)
                    img, ida_mat = self.img_transform(img, resize_dims, crop, flip, rotate)
                    images.append(img)
                    ida_mats.append(ida_mat)
        extrinsic = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        extrinsic[:, :3, :3] = sample["rots"]
        extrinsic[:, :3, 3] = sample["trans"]
        intrinsic = sample['intrins']
        ctr_points = np.zeros((self.max_instances, max(self.max_pieces) * max(self.num_degree) + 1, 2), dtype=np.float)
        ins_labels = np.zeros((self.max_instances, 3), dtype=np.int16) - 1
        for ins_id, ctr_info in enumerate(sample['ctr_points']):
            cls_id = int(ctr_info['type'])
            ctr_pts_raw = np.array(ctr_info['pts'])
            max_points = self.max_pieces[cls_id] * self.num_degree[cls_id] + 1
            num_points = max_points if max_points <= ctr_pts_raw.shape[0] else ctr_pts_raw.shape[0]
            assert num_points >= self.num_degree[cls_id] + 1
            ctr_points[ins_id][:num_points] = np.array(ctr_pts_raw[:num_points])
            ins_labels[ins_id] = [cls_id, (num_points - 1) // self.num_degree[cls_id] - 1, num_points]
        masks = sample[self.mask_key]
        if flip:
            new_order = [2, 1, 0, 5, 4, 3]
            img_key_list = [self.img_key_list[i] for i in new_order]
            images = [images[i] for i in new_order]
            ida_mats = [ida_mats[i] for i in new_order]
            extrinsic = [extrinsic[i] for i in new_order]
            intrinsic = [intrinsic[i] for i in new_order]
            masks = [np.flip(mask, axis=1) for mask in masks]
            ctr_points = self.point_flip(ctr_points, ins_labels, self.ego_size)
        item = dict(
            images=images, targets=dict(masks=masks, points=ctr_points, labels=ins_labels),
            extrinsic=np.stack(extrinsic), intrinsic=np.stack(intrinsic), ida_mats=np.stack(ida_mats),
            extra_infos=dict(token=token, img_key_list=self.img_key_list, map_size=self.ego_size, do_flip=flip)
        )
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def __len__(self):
        return len(self.tokens)

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        resize_dims = w, h = self.ida_conf["resize_dims"]
        crop = (0, 0, w, h)
        if self.ida_conf["up_crop_ratio"] > 0:
            crop = (0, int(self.ida_conf["up_crop_ratio"] * h), w, h)
        flip, color, rotate_ida = False, False, 0
        if self.split_mode == "train":
            if self.ida_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            if self.ida_conf["rot_lim"]:
                assert isinstance(self.ida_conf["rot_lim"], (tuple, list))
                rotate_ida = np.random.uniform(*self.ida_conf["rot_lim"])
        return resize_dims, crop, flip, rotate_ida

    def img_transform(self, img, resize_dims, crop, flip, rotate):
        img = Image.fromarray(img)
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        W, H = img.size
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        scales = torch.tensor([resize_dims[0] / W, resize_dims[1] / H])
        ida_rot *= torch.Tensor(scales)
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = ida_rot.new_zeros(3, 3)
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return np.asarray(img), ida_mat

    @staticmethod
    def point_flip(points, labels, map_shape):

        def _flip(pts):
            pts[:, 0] = map_shape[1] - pts[:, 0]
            return pts.copy()

        points_ret = deepcopy(points)
        for ins_id in range(points.shape[0]):
            end = labels[ins_id, 2]
            points_ret[ins_id][:end] = _flip(points[ins_id][:end])

        return points_ret

    @staticmethod
    def get_rot(h):
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
