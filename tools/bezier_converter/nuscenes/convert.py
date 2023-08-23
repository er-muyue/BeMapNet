import os
import argparse
import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from rasterize import RasterizedLocalMap
from vectorize import VectorizedLocalMap


class NuScenesDataset(Dataset):
    def __init__(self, version, dataroot, xbound=(-30., 30., 0.15), ybound=(-15., 15., 0.15)):
        super(NuScenesDataset, self).__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, idx):
        record = self.nusc.sample[idx]
        location = self.nusc.get('log', self.nusc.get('scene', record['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose',
                                 self.nusc.get('sample_data', record['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        imgs, trans, rots, intrins = self.get_data_info(record)
        return imgs, np.stack(trans), np.stack(rots), np.stack(intrins), vectors

    def get_data_info(self, record):
        imgs, trans, rots, intrins = [], [], [], []
        for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
            samp = self.nusc.get('sample_data', record['data'][cam])
            imgs.append(samp['filename'])
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(sens['translation'])
            rots.append(Quaternion(sens['rotation']).rotation_matrix)
            intrins.append(sens['camera_intrinsic'])
        return imgs, trans, rots, intrins


class NuScenesSemanticDataset(NuScenesDataset):
    def __init__(self, version, dataroot, xbound, ybound, thickness, num_degrees, max_channel=3):
        super(NuScenesSemanticDataset, self).__init__(version, dataroot, xbound, ybound)
        self.raster_map = RasterizedLocalMap(self.patch_size, self.canvas_size, num_degrees, max_channel, thickness)

    def __getitem__(self, idx):
        record = self.nusc.sample[idx]
        location = self.nusc.get('log', self.nusc.get('scene', record['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', record['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        imgs, trans, rots, intrins = self.get_data_info(record)
        semantic_masks, instance_masks, instance_vec_points, instance_ctr_points = \
            self.raster_map.convert_vec_to_mask(vectors)
        return imgs, np.stack(trans), np.stack(rots), np.stack(intrins), semantic_masks, instance_masks, \
               vectors, instance_vec_points, instance_ctr_points


def main():

    parser = argparse.ArgumentParser(description='Bezier GT Generator.')
    parser.add_argument('-d', '--data_root', type=str, default='./data')
    parser.add_argument('-n', '--data_name', type=str, default='bemapnet')
    parser.add_argument('-v', '--version', nargs='+', type=str, default=['v1.0-test', 'v1.0-trainval'])
    parser.add_argument("--num_degrees", nargs='+', type=int, default=[2, 1, 3])
    parser.add_argument("--thickness", nargs='+', type=int, default=[1, 8])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    args = parser.parse_args()

    n_classes = len(args.num_degrees)  # 0 --> divider(d=2),  1 --> crossing(d=1),  2--> contour(d=3)
    save_dir = os.path.join(args.data_root, 'customer', args.data_name)
    os.makedirs(save_dir, exist_ok=True)
    for version in args.version:
        dataset = NuScenesSemanticDataset(
            version, args.data_root, args.xbound, args.ybound, args.thickness, args.num_degrees, max_channel=n_classes)
        for idx in tqdm(range(dataset.__len__())):
            file_path = os.path.join(save_dir, dataset.nusc.sample[idx]['token'] + '.npz')
            if os.path.exists(file_path):
                continue
            item = dataset.__getitem__(idx)
            np.savez_compressed(
                file_path, image_paths=np.array(item[0]), trans=item[1], rots=item[2], intrins=item[3],
                semantic_mask=item[4][0], instance_mask=item[5][0], instance_mask8=item[5][1],
                ego_vectors=item[6], map_vectors=item[7], ctr_points=item[8]
            )


if __name__ == '__main__':
    main()
