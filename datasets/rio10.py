"""Dataset interface for rio10."""
from pathlib import Path

import yaml
import numpy as np
import cv2
from torch.utils.data import Dataset

import datasets

from .utils import DataCache, to_tensor_query, get_coord, data_aug, to_tensor


class Rio10Dataset(Dataset):
    """Dataset class for rio10."""

    def __init__(self, data_path, split='train', aug=True, seq_idx="01", **kwargs):
        super(Rio10Dataset, self).__init__()
        print("Using qike's implementation.")
        self.data_path = Path(data_path)
        self.split = split
        self.aug = aug
        self.scene_id = self.data_path.name[-2:]
        self.seq_idx = seq_idx
        self.img_shape = (960, 540, 3)

        if self.split == 'train':
            self.seq_idx = "01"
        elif self.split in ('validation', 'test'):
            self.seq_idx = "02"
        elif self.split == "eval":
            if int(seq_idx) <= 2:
                raise ValueError("Invalid seq idx for evaluation!")
        self.seq_path = self.data_path / f"seq{self.scene_id}" / f"seq{self.scene_id}_{self.seq_idx}"

        self.intrinsics = self.read_intrinsics()
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)
        self.intrinsics_color = self.intrinsics
        self.centers = np.load(str(self.data_path / f"seq{self.scene_id}" / "centers.npy"))

        self.len = len(list(self.seq_path.glob("*color.jpg")))
        # self.target_shape = (640, 480)
        self.target_shape = (self.img_shape[0] // 32 * 32, self.img_shape[1] // 32 * 32)
        self.offset = None  # (0, 0)

    def read_intrinsics(self):
        intrinsics_path = self.seq_path / "camera.yaml"
        with open(str(intrinsics_path), 'r') as f:
            data = yaml.safe_load(f)
            model = data['camera_intrinsics']['model']
            intrinsic_matrix = [
                [model[0], 0, model[2]],
                [0, model[1], model[3]],
                [0, 0, 1]
            ]
            intrinsic_matrix = np.array(intrinsic_matrix)
        return intrinsic_matrix

    def get_frame_by_idx(self, idx):
        frame = {}
        for ext_name in ['color.jpg', 'depth.png', 'label.png', 'pose.txt']:
            tag = ext_name.split('.')[0]
            if self.split == "validation" and tag == "depth":
                ext_name = "rendered.depth.png"
            frame[tag] = f"frame-{str(idx).zfill(6)}.{ext_name}"
        return frame
    
    def read_img(self, img_filename):
        img = cv2.imread(str(self.seq_path / img_filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def read_depth(self, depth_filename):
        depth = cv2.imread(str(self.seq_path / depth_filename), -1)
        return depth
    
    def read_label(self, label_filename):
        label = cv2.imread(str(self.seq_path / label_filename), -1)
        return label
    
    def read_pose(self, pose_filename):
        pose = np.loadtxt(str(self.seq_path / pose_filename))
        if self.split == "train":
            pose[0:3, 3] = pose[0:3, 3] * 1000
        return pose
    
    def crop(self, matrix, offset=None):
        if self.offset or offset:
            offset_1, offset_2 = self.offset or offset
        else:
            offset_1 = np.random.randint(0, self.img_shape[0] - self.target_shape[0] + 1)
            offset_2 = np.random.randint(0, self.img_shape[1] - self.target_shape[1] + 1)
        shape = self.target_shape
        return (offset_1, offset_2), matrix[offset_1:shape[0]+offset_1, offset_2:shape[1]+offset_2]

    def __len__(self):
        # return 10
        return self.len // 1
    
    def __getitem__(self, idx):
        frame = self.get_frame_by_idx(idx)
        img = self.read_img(frame['color'])
        depth = self.read_depth(frame['depth'])
        label = self.read_label(frame['label'])
        pose = self.read_pose(frame['pose'])

        if self.split == "test":
            _, img = self.crop(img, (0, 14))
            img, pose = to_tensor_query(img, pose)
            return img, pose
        
        center_coord = self.centers[np.reshape(label, (-1))-1, :]
        center_coord = np.reshape(center_coord, (960, 540, 3)) * 1000

        depth[depth == 65535] = 0
        depth = depth * 1.0
        coord, mask = get_coord(depth, pose, self.intrinsics_inv)
        img, coord, center_coord, mask, label = data_aug(
            img=img,
            coord=coord,
            ctr_coord=center_coord,
            mask=mask,
            lbl=label,
            aug=self.aug
        )

        # get relative coord to center of sub regions
        coord = coord - center_coord

        offset, img = self.crop(img)
        _, coord = self.crop(coord, offset)
        _, mask = self.crop(mask, offset)
        _, label = self.crop(label, offset)

        coord = coord[4::8, 4::8, :]
        mask = mask[4::8,4::8].astype(np.float16)
        label = label[4::8,4::8].astype(np.float16)

        label_1 = (label - 1) // 25
        label_2 = ((label - 1) % 25)

        img, coord, mask, label_1, label_2, label_1_oh, label_2_oh = to_tensor(
            img, coord, mask, label_1, label_2, 25
        )
        return img, coord, mask, label_1, label_2, label_1_oh, label_2_oh
    
    def generate_label(self, idx):
        print(f"Generating label for the {idx}-th frame.")
        if self.split == "train":
            print("Split is train. Skipping...")
            return
        frame = self.get_frame_by_idx(idx)
        if (self.seq_path / frame['label']).is_file():
            print(f"{self.seq_path / frame['label']} exists. Skipping...")
            return
        depth = self.read_depth(frame['depth'])
        pose = self.read_pose(frame['pose'])
        depth[depth == 65535] = 0
        depth = depth * 1.0
        coord, mask = get_coord(depth, pose, self.intrinsics_inv)
        label = np.tile(coord[:, :, :, np.newaxis], (1, 1, 1, self.centers.shape[0])) - self.centers.T * 1000
        label = np.sum(label ** 2, axis=-2)
        label = np.argmin(label, axis=-1) + 1
        label = label * mask
        cv2.imwrite(str(self.seq_path / frame['label']), label)


if __name__ == "__main__":
    dataset = Rio10Dataset("data/rio10/scene01")
    dataset.generate_label()
