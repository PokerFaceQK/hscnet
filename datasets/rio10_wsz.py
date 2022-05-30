import os
import random
from imgaug.augmenters import segmentation
import numpy as np
import cv2
from torch.utils import data
from skimage.segmentation import slic

from .utils import *


class RIOScenes(data.Dataset):
    def __init__(self, root, dataset='RIO10', scene='scene01/seq01', split='train',
                    model='hscnet', aug='True'):
        self.intrinsics_color = np.array([[756.0, 0.0,     270.4],
                       [0.0,     756.8, 492.9],
                       [0.0,     0.0,  1.0]])
                       
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        # self.add_unlabel = unlabel
        # self.add_unlabel_noise = False
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'RIO10')
        self.scene = scene
        self.train_scene = 'seq01_01'
        self.eval_scene = 'seq01_02'
        if self.dataset == 'RIO10':
            self.centers = np.load(os.path.join(self.root, scene,
                            'centers.npy'))
        else: 
            self.scenes = ['apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
            self.transl = [[0,-20,0],[0,-20,0],[20,0,0],[20,0,0],[25,0,0],
                    [20,0,0],[-20,0,0],[-25,5,0],[-20,0,0],[-20,-5,0],[0,20,0],
                    [0,20,0]]
            if self.dataset == 'i12S':
                self.ids = [0,1,2,3,4,5,6,7,8,9,10,11]
            else:
                self.ids = [7,8,9,10,11,12,13,14,15,16,17,18]
            self.scene_data = {}
            for scene, t, d in zip(self.scenes, self.transl, self.ids):
                self.scene_data[scene] = (t, d, np.load(os.path.join(self.root,
                    scene,  'centers.npy')))

        self.split = split
        self.obj_suffixes = ['.color.jpg', '.pose.txt', '.depth.png', 
                '.label.png']
        self.obj_keys = ['color', 'pose', 'depth', 'label']
        
        if self.dataset == 'RIO10' or self.split == 'test':
            with open(os.path.join(self.root, self.scene, 
                    '{}{}'.format(self.split, '.txt')), 'r') as f:
                self.frames = f.readlines()
        else:
            self.frames = []
            for scene in self.scenes:
                with open(os.path.join(self.root, scene, 
                        '{}{}'.format(self.split, '.txt')), 'r') as f:
                    frames = f.readlines()
                    frames = [scene + ' ' + frame for frame in frames ]
                self.frames.extend(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')

        if self.dataset != 'RIO10' and self.split == 'train':
            scene, frame = frame.split(' ')
            centers = self.scene_data[scene][2] 
        else: 
            scene = self.scene
            if self.split == 'train':
                centers = self.centers
        
        obj_files = ['{}{}'.format(frame, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        if self.split == 'train':
            obj_files_full = [os.path.join(self.root, scene, self.train_scene, 
                        obj_file) for obj_file in obj_files]
        if self.split == 'test':
            obj_files_full = [os.path.join(self.root, scene, self.eval_scene, 
                        obj_file) for obj_file in obj_files]
       
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data

        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.copyMakeBorder(img,0,0,0,4,cv2.BORDER_CONSTANT,value=0) # padding to 960*544

        pose = np.loadtxt(objs['pose'])
        if self.dataset != 'RIO10' and (self.model != 'hscnet' \
                        or self.split == 'test'):
            pose[0:3,3] = pose[0:3,3] + np.array(self.scene_data[scene][0])

        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose
        
        lbl = cv2.imread(objs['label'],-1)
        lbl = cv2.copyMakeBorder(lbl,0,0,0,4,cv2.BORDER_CONSTANT,value=1) # padding to 960*544

        ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
        ctr_coord = np.reshape(ctr_coord,(960,544,3)) * 1000

        depth = cv2.imread(objs['depth'],-1)
        depth = cv2.copyMakeBorder(depth,0,0,0,4,cv2.BORDER_CONSTANT,value=0) # padding to 960*544

        pose[0:3,3] = pose[0:3,3] * 1000

        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        img, coord, ctr_coord, mask, lbl = data_aug(img, coord, ctr_coord, 
                mask, lbl, self.aug)

        if self.model == 'hscnet':
            coord = coord - ctr_coord
               
        coord = coord[4::8,4::8,:]
        mask = mask[4::8,4::8].astype(np.float16)
        lbl = lbl[4::8,4::8].astype(np.float16)
       
        lbl_1 = (lbl - 1) // 25
        lbl_2 = ((lbl - 1) % 25) 
        # if self.add_unlabel:
        #     unlabel_mask = np.where(lbl == 626)
        #     lbl_1[unlabel_mask] = 25
        #     lbl_2[unlabel_mask] = 25

 
        #     unlbl_mask = np.ones_like(mask)
        #     unlbl_mask[unlabel_mask] = 0
        # else:
        #     unlbl_mask = np.ones_like(mask)
        if  self.dataset=='RIO10':
            N1=25
        img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh = to_tensor(img, 
                    coord, mask,lbl_1, lbl_2, N1)

        return img, coord, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh