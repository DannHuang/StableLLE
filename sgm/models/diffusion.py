import math
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import normalize

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log


class IRDiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        restorer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        struct_cond_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        degradation_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        freeze_unet: bool = True,
        restorer_w: float = 1.0,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.freeze_unet = freeze_unet
        model = self._init_model(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self.degradation_config = degradation_config
        self._init_first_stage(first_stage_config)
        # self._init_first_restorer(restorer_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.restorer_w = restorer_w

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
    
    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_model(self, config):
        if self.freeze_unet:
            model = instantiate_from_config(config).eval()
            for name, param in model.named_parameters():
                if 'spade' not in name and 'lq_encoder' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            model = instantiate_from_config(config)
        return model
    
    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def _init_first_restorer(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.restorer = model

    @torch.no_grad()
    def get_input(self, batch, bs=None, val=False, text_cond=[''], resize_lq=True):

        """
            output: {
                gt: input,
                gt_path: input dir,
                txt: text condition,
                original_size_as_tuple: raw input size,
                target_size_as_tuple: resize size,
                crop_coords_top_left: crop coordinate,
                lq: degradated input,
                cf_out: CodeFormer output,
                struct_cond_dict: CodeFormer feature-dict
                }
        """
        val = not self.training
        if not hasattr(self, 'jpeger'):
            self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        if not hasattr(self, 'usm_sharpener'):
            self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        
        im_gt = batch['gt'].cuda()
        im_gt = self.usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch.pop('kernel1').cuda()
        kernel2 = batch.pop('kernel2').cuda()
        sinc_kernel = batch.pop('sinc_kernel').cuda()

        ori_h, ori_w = im_gt.size()[2:4]        # 512x512

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.degradation_config['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.degradation_config['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degradation_config['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = torch.nn.functional.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.degradation_config['gray_noise_prob']
        if random.random() < self.degradation_config['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.degradation_config['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.degradation_config['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation_config['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.degradation_config['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.degradation_config['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.degradation_config['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degradation_config['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = torch.nn.functional.interpolate(
                out,
                size=(int(ori_h / self.degradation_config['sf'] * scale),
                    int(ori_w / self.degradation_config['sf'] * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.degradation_config['gray_noise_prob2']
        if random.random() < self.degradation_config['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.degradation_config['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.degradation_config['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = torch.nn.functional.interpolate(
                    out,
                    size=(ori_h // self.degradation_config['sf'],
                        ori_w // self.degradation_config['sf']),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation_config['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation_config['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = torch.nn.functional.interpolate(
                    out,
                    size=(ori_h // self.degradation_config['sf'],
                        ori_w // self.degradation_config['sf']),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        gt_size = self.degradation_config['gt_size']           # 512x512
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.degradation_config['sf'])
        self.lq, self.gt = im_lq, im_gt

        if resize_lq:                   # resize back to 512x512
            self.lq = torch.nn.functional.interpolate(
                    self.lq,
                    size=(self.gt.size(-2),
                        self.gt.size(-1)),
                    mode='bicubic',
                    )

        if random.random() < self.degradation_config['no_degradation_prob'] or torch.isnan(self.lq).any():
            self.lq = self.gt

        # training pair pool
        if not val and not self.degradation_config['random_size']:
            self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq = self.lq*2 - 1.0
        self.gt = self.gt*2 - 1.0

        if self.degradation_config['random_size']:
            self.lq, self.gt = self.randn_cropinput(self.lq, self.gt)

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq
        y = self.gt
        if bs is not None:
            x = x[:bs]
            y = y[:bs]
        x = x.to(self.device)
        y = y.to(self.device)
        # # Encode LQ
        # z = self.encode_first_stage(x)
        # # Encode HQ
        # z_gt = self.encode_first_stage(y)

        # process text input
        xc = None
        while len(text_cond) < x.size(0):
            text_cond.append(text_cond[-1])
        if len(text_cond) > x.size(0):
            text_cond = text_cond[:x.size(0)]
        assert len(text_cond) == x.size(0)

        batch['gt'] = y
        batch['lq'] = x
        batch['txt'] = text_cond
        cropped_face_t = (x+1.0)/2.0
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cf_recon, struct_c = self.restorer(cropped_face_t, w=self.restorer_w, adain=False, hq_code=False)
        batch['cf_out'] = cf_recon
        batch['struct_cond_dict'] = struct_c

        return batch

    def randn_cropinput(self, lq, gt, base_size=[64, 128, 256, 512]):
        cur_size_h = random.choice(base_size)
        cur_size_w = random.choice(base_size)
        init_h = lq.size(-2)//2
        init_w = lq.size(-1)//2
        lq = lq[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
        gt = gt[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
        assert lq.size(-1)>=64
        assert lq.size(-2)>=64
        return [lq, gt]
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch, taken from Real-ESRGAN:
        https://github.com/xinntao/Real-ESRGAN

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if b == self.degradation_config['batch_size']:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.degradation_config.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_ptr = self.queue_ptr + b
    
    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        batch = self.get_input(batch)
        x = self.encode_first_stage(batch['gt'])
        batch['struct_cond'] = self.encode_first_stage(batch['cf_out'])
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **additional_model_inputs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)     # init latent of shape [N, 4, 64, 64]

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **additional_model_inputs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys   # Unconditional Guidance set to zero
        log = dict()

        batch = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}
        
        input = batch['gt']
        lq = batch['lq']

        N = min(input.shape[0], N)
        input = input.to(self.device)[:N]
        lq = lq.to(self.device)[:N]
        log["inputs"] = lq
        log["gt"] = input
        z_gt = self.encode_first_stage(input)
        z_cf = self.encode_first_stage(batch['cf_out'][:N])
        log["cf"] = batch['cf_out'][:N]
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z_gt.shape[1:], uc=uc, batch_size=N, struct_cond=z_cf, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log