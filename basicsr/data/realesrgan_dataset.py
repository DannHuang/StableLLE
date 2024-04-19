import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, **opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.crop_size = opt.get('crop_size', 512)
        self.gaussian_crop = opt.get('gaussian_crop', False)
        if self.gaussian_crop:
            self.gaussian_w = opt.get('gaussian_w')
            self.gaussian_h = opt.get('gaussian_h')
        self.org_prob = opt.get('org_prob', 0)
        self.use_degrad = opt.get('use_degrad', True)
        # assert len(opt['dataset_prob']) == len(opt['dataset_configs'])
        print(f"mixing {len(opt['dataset_configs'])} datasets.")

        # Mixing datasets configs
        self.paths = []
        self.resize_range = []
        self.bins = []
        for config in opt['dataset_configs']:
            if 'image_type' not in config:
                config['image_type'] = 'png'
            if 'resize_range' not in config:
                config['resize_range'] = []
            
            # support multiple type of data: file path and meta data, remove support of lmdb
            local_paths = []
            if 'meta_info' in config:
                with open(self.config['meta_info']) as fin:
                        paths = [line.strip().split(' ')[0] for line in fin]
                        self.paths = [v for v in paths]
                if 'meta_num' in config:
                    self.paths = sorted(self.paths)[:config['meta_num']]
            elif 'gt_path' in config:
                if isinstance(config['gt_path'], str):
                    local_paths.extend(sorted([str(x) for x in Path(config['gt_path']).glob('**/*.'+config['image_type'])]))
                else:
                    local_paths.extend(sorted([str(x) for x in Path(config['gt_path'][0]).glob('**/*.'+config['image_type'])]))
                    if len(config['gt_path']) > 1:
                        for i in range(len(config['gt_path'])-1):
                            local_paths.extend(sorted([str(x) for x in Path(config['gt_path'][i+1]).glob('**/*.'+config['image_type'])]))
            elif 'imagenet_path' in config:
                class_list = os.listdir(config['imagenet_path'])
                for class_file in class_list:
                    local_paths.extend(sorted([str(x) for x in Path(os.path.join(config['imagenet_path'], class_file)).glob('**/*.'+'JPEG')]))
            elif 'face_gt_path' in config:
                if isinstance(config['face_gt_path'], str):
                    face_list = sorted([str(x) for x in Path(config['face_gt_path']).glob('**/*.[jpJP][pnPN]*[gG]')])
                    local_paths.extend(face_list[:config['num_face']])
                    # local_paths.extend(sorted([str(x) for x in Path(config['face_gt_path']).glob('**/*.[jpJP][pnPN]*[gG]')]))
                else:
                    face_list = sorted([str(x) for x in Path(config['face_gt_path'][0]).glob('**/*.[jpJP][pnPN]*[gG]')])
                    if len(config['face_gt_path']) > 1 and len(face_list)<config['num_face']:
                        for i in range(len(config['face_gt_path'])-1):
                            face_list.extend(sorted([str(x) for x in Path(config['face_gt_path'][i+1]).glob('**/*.[jpJP][pnPN]*[gG]')])[:(config['num_face']-len(face_list))])
                    local_paths.extend(face_list)

            # limit number of pictures for test
            if config.get('mul_num', None):
                local_paths = local_paths * config['mul_num']
            if config.get('num_pic', None):
                if 'val' or 'test' in opt:
                    random.shuffle(local_paths)
                    local_paths = local_paths[-config['num_pic']:]
                else:
                    local_paths = local_paths[-config['num_pic']:]
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)
            print(f"get {len(local_paths)} images from {config['gt_path']}")
            self.paths.append(local_paths)
            self.resize_range.append(config['resize_range'])
            self.bins.append(len(local_paths))
        for i in range(len(self.bins)):
            if i>0: self.bins[i] = self.bins[i-1]+self.bins[i]
        print(f"Mixed-dataset distribution: {self.bins}")
        
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        # self.use_jpeg = opt.get('use_jpeg', True)
        # self.use_usm = opt.get('usm_sharpener', True)
        # if self.use_jpeg:
        #     self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        # if self.use_usm:
        #     self.sharpener = USMSharp().cuda()  # do usm sharpening

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        dataset_index = 0
        while dataset_index < len(self.bins):
            if index < self.bins[dataset_index]:
                break
            else: dataset_index+=1
        if dataset_index==0:
            gt_path = self.paths[0][index]
        else: gt_path = self.paths[dataset_index][index-self.bins[dataset_index-1]]
        resize_range = self.resize_range[dataset_index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                dataset_index = 0
                while dataset_index < len(self.bins):
                    if index < self.bins[dataset_index]:
                        break
                    else: dataset_index+=1
                if dataset_index==0:
                    gt_path = self.paths[0][index]
                else: gt_path = self.paths[dataset_index][index-self.bins[dataset_index-1]]
                resize_range = self.resize_range[dataset_index]
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
            dataset_index = 0
            while dataset_index < len(self.bins):
                if index < self.bins[dataset_index]:
                    break
                else: dataset_index+=1
            if dataset_index==0:
                gt_path = self.paths[0][index]
            else: gt_path = self.paths[dataset_index][index-self.bins[dataset_index-1]]
            resize_range = self.resize_range[dataset_index]

            time.sleep(0.1)  # sleep 1s for occasional server congestion
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_size = os.path.getsize(gt_path)
            img_size = img_size/1024

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad
        return_d = dict()
        h, w = img_gt.shape[0], img_gt.shape[1]
        return_d["original_size_as_tuple"] = torch.tensor([h, w])
        crop_pad_size = self.crop_size
        # input too small, pad
        if h < crop_pad_size or w < crop_pad_size:
            raise ValueError(f"Input images with size ({h}, {w}) smaller than "
                             f"training resolution will lead to degraded performance.")
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # input too large, downsize+crop
        if h > crop_pad_size or w > crop_pad_size:
            # Apply random downsize to buckets to increase robustness
            if np.random.uniform() > self.opt['org_prob'] and len(resize_range)>0: 
                if h > resize_range[0] or w > resize_range[0]:
                    resize_idx = random.randint(0, len(resize_range)-1)
                    while resize_range[resize_idx] > h or resize_range[resize_idx] > w:
                        resize_idx = random.randint(0, resize_idx-1)
                    resize_size = resize_range[resize_idx]
                    scale = max(h/resize_size, w/resize_size)
                    w = int(w / scale)
                    h = int(h / scale)
                    img_gt = cv2.resize(img_gt, (w, h), interpolation=cv2.INTER_AREA)
            return_d['target_size_as_tuple'] = torch.tensor([img_gt.shape[0], img_gt.shape[1]])
            # size = min(jpg.shape[1], jpg.shape[2])
            # delta_h = jpg.shape[1] - size
            # delta_w = jpg.shape[2] - size
            # assert not all(
            #     [delta_h, delta_w]
            # )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
            # then, randomly choose top and left coordinates
            if self.gaussian_crop:
                top = np.clip(np.random.normal(*self.gaussian_h), -3*self.gaussian_h[1], 3*self.gaussian_h[1])
                top = (top+3*self.gaussian_h[1])/6/self.gaussian_h[1]
                top = max(0, min(int(top*h)-crop_pad_size//2,h-crop_pad_size-1))
                left = np.clip(np.random.normal(*self.gaussian_w), -3*self.gaussian_w[1], 3*self.gaussian_w[1])
                left = (left+3*self.gaussian_w[1])/6/self.gaussian_w[1]
                left = max(0, min(int(left*w)-crop_pad_size//2,w-crop_pad_size-1))
            else:
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
            # top = (int(h/scale) - crop_pad_size) // 2 -1
            # left = (int(w/scale) - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            return_d["crop_coords_top_left"] = torch.tensor([top, left])
        if self.use_degrad is False:
            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            return {'gt': img_gt, 'gt_path': gt_path}

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        return_d.update({'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path})
        return return_d
        

    def __len__(self):
        return self.bins[-1]
