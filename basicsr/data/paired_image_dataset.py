import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_meta_info_file_2
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import cv2
import numpy as np
import random


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, **opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.paths = []
        self.bins = []
        for config in opt['dataset_configs']:
            if 'image_type' not in config:
                config['image_type'] = 'png'
            
            # support multiple type of data: file path and meta data, remove support of lmdb
            lq_folder = os.path.join(config['root'], 'Low')
            gt_folder = os.path.join(config['root'], 'Normal')
            filename_tmpl = config.get('filename_tmpl', '{}')
            local_paths=paired_paths_from_folder([lq_folder, gt_folder], ['lq', 'gt'], filename_tmpl)

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
            print(f"get {len(local_paths)} images from {config['root']}")
            self.paths.extend(local_paths)
            self.bins.append(len(local_paths))

        for i in range(len(self.bins)):
            if i>0: self.bins[i] = self.bins[i-1]+self.bins[i]
        print(f"Mixed-dataset distribution: {self.bins}")

        # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        h, w = img_gt.shape[0:2]
        # pad
        if h < self.opt['gt_size'] or w < self.opt['gt_size']:
            pad_h = max(0, self.opt['gt_size'] - h)
            pad_w = max(0, self.opt['gt_size'] - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        if h > self.opt['gt_size'] or w > self.opt['gt_size']:
            # size = min(jpg.shape[1], jpg.shape[2])
            # delta_h = jpg.shape[1] - size
            # delta_w = jpg.shape[2] - size
            # assert not all(
            #     [delta_h, delta_w]
            # )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
            # then, randomly choose top and left coordinates
            top = random.randint(0, h - self.opt['gt_size'])
            left = random.randint(0, w - self.opt['gt_size'])
            # center crop
            # top = (int(h/scale) - crop_pad_size) // 2 -1
            # left = (int(w/scale) - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + self.opt['gt_size'], left:left + self.opt['gt_size'], ...]
            img_lq = img_lq[top:top + self.opt['gt_size'], left:left + self.opt['gt_size'], ...]


        # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
