import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
import torchvision
from PIL import Image
# from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sgm.util import instantiate_from_config
from omegaconf import OmegaConf

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    # model.eval()
    return model

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    # @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


class MyDataset(Dataset):
    def __init__(
        self,
        data_root=''
    ):
        self.data_root=data_root
        self.input_paths=self._get_image_paths(self.data_root+'/input')
        self.target_paths=self._get_image_paths(self.data_root+'/target')

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        
        prompt=" "
        
        source = cv2.imread(self.input_paths[idx])
        target = cv2.imread(self.target_paths[idx])

        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))
        # cv2.imwrite('/data3/jinhongbo/StableLLE/outputs/source.png',source)
        # cv2.imwrite('/data3/jinhongbo/StableLLE/outputs/target.png',target)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return dict(png=target, txt=prompt, hint=source)

    def _get_image_paths(self,data_root):
        image_paths = []
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)
        return image_paths

# Configs
resume_path = '/share/huangrenyuan/zoo/sd/v2-1_512-ema-pruned.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

config = OmegaConf.load('/data/huangrenyuan/projects/StableLLE/configs/LLIE/train.yaml')
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = load_model_from_config(
    config=config,
    ckpt=resume_path,
    verbose=True)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

max_epoch=2

# Misc
dataset = MyDataset('/share/huangrenyuan/dataset/lol_dataset/our485')
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=logger_freq)

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='/share/huangrenyuan/logs/StabLLE',
    filename='{epoch}-{val_loss:.2f}',  # 根据需要自定义文件名格式
    monitor='val_loss',
    mode='min',
    save_top_k=1,  # 只保存最佳的模型
)


trainer = pl.Trainer(
    accelerator='gpu', 
    devices=4,
    precision=32, 
    # callbacks=[checkpoint_callback],
    max_epochs=max_epoch
)


# # Train!
trainer.fit(model, dataloader)


output_path='/share/huangrenyuan/logs/StableLLE/finetune_.ckpt'
torch.save(model.state_dict(), output_path)

