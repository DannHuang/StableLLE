from typing import Sequence, Dict, Union
import os
import math
import time
import random
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

import torch.utils.data as data

from basicsr.data.degradations import random_add_gaussian_noise, random_mixed_kernels, random_add_jpg_compression
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.data.transforms import augment

class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        opt,
    ):
        super(CodeformerDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
        if 'imagenet_path' in opt:
            class_list = os.listdir(opt['imagenet_path'])
            for class_file in class_list:
                self.paths.extend(sorted([str(x) for x in Path(os.path.join(opt['imagenet_path'], class_file)).glob('*.'+'JPEG')]))
        if 'face_gt_path' in opt:
            if isinstance(opt['face_gt_path'], str):
                face_list = sorted([str(x) for x in Path(opt['face_gt_path']).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
            else:
                face_list = sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])
                if len(opt['face_gt_path']) > 1 and len(face_list)<opt['num_face']:
                    for i in range(len(opt['face_gt_path'])-1):
                        face_list.extend(sorted([str(x) for x in Path(opt['face_gt_path'][i+1]).glob('*.'+opt['image_type'])])[:(opt['num_face']-len(face_list))])
                self.paths.extend(face_list)

        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'crop_type' in opt:
            assert opt['crop_type'] in ["none", "center", "random"]
            self.crop_type = opt['crop_type']
        else:
            self.crop_type = "random"
        self.use_hflip = opt['use_hflip']
        # degradation configurations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

    def __getitem__(self, index: int):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        # filter the dataset and remove images with too low quality
        img_size = os.path.getsize(gt_path)
        img_size = img_size/1024

        while img_gt.shape[0] * img_gt.shape[1] < 384*384 or img_size<100:
            index = random.randint(0, self.__len__()-1)
            gt_path = self.paths[index]

            time.sleep(0.1)  # sleep 1s for occasional server congestion
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_size = os.path.getsize(gt_path)
            img_size = img_size/1024

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            if self.crop_type == "center":
                top = (h - crop_pad_size) // 2
                left = (w - crop_pad_size) // 2
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            elif self.crop_type == "random":
                # randomly choose top and left coordinates
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                # top = (h - crop_pad_size) // 2 -1
                # left = (w - crop_pad_size) // 2 -1
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            else:
                assert img_gt.shape[:2] == (self.crop_size, self.crop_size)
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        return_d = {'gt': img_gt, 'kernel': kernel, 'gt_path': gt_path}
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB
        # # target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # target = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        # # BGR to RGB, [0, 1]
        # # source = img_lq[..., ::-1].astype(np.float32)
        # source = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        # return_d = {'lq': source, 'gt': target}
        return return_d

    def __len__(self):
        return len(self.paths)