"""make variations of input image"""

import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torchvision.transforms.functional import normalize
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from sgm.util import instantiate_from_config
import math
import copy
import torch.nn.functional as F
import cv2
from scripts.util_images import ImageSpliterTh
from scripts.util_images import wavelet_reconstruction, adaptive_instance_normalization

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

# def chunk(it, size):
# 	it = iter(it)
# 	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	keys = list(sd.keys())
	ignore_keys = config.model.params.get('ignore_keys', [])
	for k in keys:
		for ik in ignore_keys:
			if k.startswith(ik):
				print("Deleting key {} from state_dict.".format(k))
				del sd[k]
	model = instantiate_from_config(config.model)
	# m, u = model.load_state_dict(sd, strict=False)
	# print("="*32)
	# print(f"Resotre from ckpt {ckpt} with {len(m)} missing keys and {len(u)} unexpected keys.")
	# if len(m) > 0 and verbose:
	# 	print("missing keys:")
	# 	print(m)
	# if len(u) > 0 and verbose:
	# 	print("unexpected keys:")
	# 	print(u)
	# print("="*32)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload"
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload"
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=2,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="./stablesr_000117.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--tile_overlap",
		type=int,
		default=32,
		help="tile overlap size",
	)
	parser.add_argument(
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
	parser.add_argument(
		"--strength",
		type=float,
		default=0.75,
		help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
	)
	parser.add_argument(
		"--use_negative_prompt",
		action='store_true',
		help="if enabled, save inputs",
	)
	parser.add_argument(
		"--use_posi_prompt",
		action='store_true',
		help="if enabled, save inputs",
	)
	parser.add_argument(
		"--vqgantile_stride",
		type=int,
		default=384,
		help="the stride for tile operation before VQGAN decoder (in pixel)",
	)
	parser.add_argument(
		"--vqgantile_size",
		type=int,
		default=512,
		help="the size for tile operation before VQGAN decoder (in pixel) subject to VRAM constraint",
	)
	parser.add_argument(
		"--gamma",
		type=float,
		default=0.0,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--use_fp16",
		action='store_true',
		help="use half precision",
	)

	opt = parser.parse_args()
	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")       # here checkpoint from CLI will loaded, make sure the time_replace parameters the same as checkpoint
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	# model = model.to(device)

	model.configs = config

	# vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	# if opt.vqgan_ckpt is not None:
	# 	print(f"Decode LDM with VQ-GAN initialized from {opt.vqgan_ckpt}.")
	# 	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	# 	vq_model = vq_model.to(device)
	# 	vq_model.decoder.fusion_w = opt.dec_w
	# else:
	# 	print(f"Decode LDM with pretrianed VAE.")

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir
	batch_size = opt.n_samples
	# Loading data
	input_video = False
	init_image_list = []
	print(f"loading inputs from {opt.init_img}")
	if isinstance(opt.init_img, str):
		if opt.init_img.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
			cur_image = load_img(os.path.join(opt.init_img)).to(device)
			# max size: 1800 x 1800 for V100
			cur_image = F.interpolate(
					cur_image,
					size=(int(cur_image.size(-2)*opt.upscale),
						int(cur_image.size(-1)*opt.upscale)),
					mode='bicubic',
					)
			cur_image = cur_image.to('cpu')
			if opt.use_fp16: cur_image.to(torch.float16)
			init_image_list.append(cur_image)
			img_list=[opt.init_img]
		elif opt.init_img.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
			input_video = True
			from basicsr.utils.video_util import VideoReader, VideoWriter
			vidreader = VideoReader(opt.init_img)
			cur_image = vidreader.get_frame()
			while cur_image is not None:
				cur_image = F.interpolate(
						cur_image,
						size=(int(cur_image.size(-2)*opt.upscale),
							int(cur_image.size(-1)*opt.upscale)),
						mode='bicubic',
						)
				cur_image = cur_image.to('cpu')
				if opt.use_fp16: cur_image.to(torch.float16)
				init_image_list.append(cur_image)
				cur_image = vidreader.get_frame()
			# audio = vidreader.get_audio()
			# fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
			# video_name = os.path.basename(opt.init_img)[:-4]
			# result_root = f'results/{video_name}_{w}'
			# input_video = True
			vidreader.close()
		else: # input img folder
			img_list_ori = os.listdir(opt.init_img)
			img_list = copy.deepcopy(img_list_ori)
			for item in img_list_ori:
				if os.path.exists(os.path.join(outpath, item)):
					img_list.remove(item)
					continue
				cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
				cur_image = cur_image.to('cpu')
				if opt.use_fp16: cur_image.to(torch.float16)
				init_image_list.append(cur_image)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				for n in trange(len(init_image_list), desc="Sampling"):
					pch_out=[]
					init_image = init_image_list[n].to(device)
					size_min = min(init_image.size(-1), init_image.size(-2))
					# max size: 1800 x 1800 for V100
					# if min(init_image.size(-2), init_image.size(-1))*opt.upscale > 1800:
					# 	print(f'input image too large, resize to 1800')
					# 	upscale = 1800/max(init_image.size(-2), init_image.size(-1))
					# else
					upsample_scale = max(opt.input_size/size_min, opt.upscale)
					init_image = F.interpolate(
							init_image,
							size=(int(init_image.size(-2)*upsample_scale),
								int(init_image.size(-1)*upsample_scale)),
							mode='bicubic',
							)
					init_image = init_image.clamp(-1.0, 1.0)
					ori_h, ori_w = init_image.shape[2:]
					if not (ori_h % 32 == 0 and ori_w % 32 == 0):
						flag_pad = True
						pad_h = ((ori_h // 32) + 1) * 32 - ori_h
						pad_w = ((ori_w // 32) + 1) * 32 - ori_w
						init_image = F.pad(init_image, pad=(0, pad_w, 0, pad_h), mode='reflect')
					else:
						flag_pad = False
					ori_size = None

					print('>>>>>>>>>>>>>>>>>>>>>>>')
					original_h, original_w = init_image.shape[2], init_image.shape[3]
					print(f"input LQ image size={init_image.size()}")
					ucg_keys=None       # TODO: add to CLI
					sampling_kwargs = {}

					if init_image.size(-1) < opt.input_size or init_image.size(-2) < opt.input_size:
						raise RuntimeError("model input size is larger than input LQ image size")
						ori_size = init_image.size()
						new_h = max(ori_size[-2], opt.input_size)
						new_w = max(ori_size[-1], opt.input_size)
						init_template = torch.zeros(1, init_image.size(1), new_h, new_w).to(init_image.device)
						init_template[:, :, :ori_size[-2], :ori_size[-1]] = init_image
					# else:
					# 	init_template = init_image
					# if cfw_decode: init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
					# else: init_latent_generator = model.encode_first_stage(init_image)
					# init_latent = model.get_first_stage_encoding(init_latent_generator)
					if init_image.shape[2] > opt.vqgantile_size or init_image.shape[3] > opt.vqgantile_size:
						im_spliter = ImageSpliterTh(init_image, opt.vqgantile_size, opt.vqgantile_stride, sf=1, gather_method='gaussian')
						conditioner_input_keys = [e.input_key for e in model.conditioner.embedders]
						if ucg_keys:
							assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
								"Each defined ucg key for sampling must be in the provided conditioner input keys,"
								f"but we have {ucg_keys} vs. {conditioner_input_keys}"
							)
						else:
							ucg_keys = conditioner_input_keys   # Unconditional Guidance set to zero
						for im_lq_pch, index_infos in im_spliter:
							seed_everything(opt.seed)
							# pch_out = torch.clamp((im_lq_pch + 1.0) / 2.0, min=0.0, max=1.0).squeeze(0)
							# pch_out = pch_out.cpu().numpy().transpose(1,2,0)*255   # b h w c
							# Image.fromarray(pch_out.astype(np.uint8)).save(os.path.join(outpath, 'pch.png'))
							init_latent = model.encode_first_stage(im_lq_pch)  # move to latent space
							cropped_face_t = (im_lq_pch+1.0)/2.0
							normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
							cf_out, struct_c = model.restorer(cropped_face_t, w=model.restorer_w, adain=False, hq_code=False)
							cf_out = cf_out.to(im_lq_pch.dtype)
							z_cf = model.encode_first_stage(cf_out)
							batch=dict()
							batch["txt"] = ['']*opt.n_samples
							batch["original_size_as_tuple"] = torch.cat([torch.tensor([original_h, original_w]).unsqueeze(0)]*opt.n_samples).to(im_lq_pch)
							batch["target_size_as_tuple"] = torch.cat([torch.tensor([original_h, original_w]).unsqueeze(0)]*opt.n_samples).to(im_lq_pch)
							batch["crop_coords_top_left"] = torch.cat([torch.tensor([index_infos[0], index_infos[2]]).unsqueeze(0)]*opt.n_samples).to(im_lq_pch)
							c, uc = model.conditioner.get_unconditional_conditioning(
								batch,
								force_uc_zero_embeddings=ucg_keys
								if len(model.conditioner.embedders) > 0
								else [],
							)
							if opt.use_negative_prompt:
								raise NotImplementedError
								negative_text_init = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)']*im_lq_pch.size(0)
								# negative_text_init = ['Bad photo.']*im_lq_pch.size(0)
								nega_semantic_c = model.cond_stage_model(negative_text_init)
							noise = torch.randn_like(init_latent)
							# TODO: If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
							# t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
							# t = t.to(device).long()
							# x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
							x_T = None
							# samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=im_lq_pch, batch_size=im_lq_pch.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)
							samples = model.sample(
								c, shape=init_latent.shape[1:], uc=uc, batch_size=opt.n_samples, struct_cond=z_cf, **sampling_kwargs
							)
							x_samples = model.decode_first_stage(samples)
							del samples
							# if opt.colorfix_type == 'adain':
							# 	x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
							# elif opt.colorfix_type == 'wavelet':
							# 	x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
							pch_out.append(torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0))
							im_spliter.update(x_samples, index_infos)
						im_sr = im_spliter.gather()
					else:
						seed_everything(opt.seed)
						conditioner_input_keys = [e.input_key for e in model.conditioner.embedders]
						if ucg_keys:
							assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
								"Each defined ucg key for sampling must be in the provided conditioner input keys,"
								f"but we have {ucg_keys} vs. {conditioner_input_keys}"
							)
						else:
							ucg_keys = conditioner_input_keys   # Unconditional Guidance set to zero
						init_latent = model.encode_first_stage(init_image)
						cropped_face_t = (init_image+1.0)/2.0
						normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
						cf_out, struct_c = model.restorer(cropped_face_t, w=model.restorer_w, adain=False, hq_code=False)
						cf_out = cf_out.to(init_image.dtype)
						z_cf = model.encode_first_stage(cf_out)
						batch=dict()
						batch["txt"] = ['']*opt.n_samples
						batch["original_size_as_tuple"] = torch.cat([torch.tensor([512, 512]).unsqueeze(0)]*opt.n_samples).to(init_image)
						batch["target_size_as_tuple"] = torch.cat([torch.tensor([512, 512]).unsqueeze(0)]*opt.n_samples).to(init_image)
						batch["crop_coords_top_left"] = torch.cat([torch.tensor([0, 0]).unsqueeze(0)]*opt.n_samples).to(init_image)
						c, uc = model.conditioner.get_unconditional_conditioning(
							batch,
							force_uc_zero_embeddings=ucg_keys
							if len(model.conditioner.embedders) > 0
							else [],
							)
						if opt.use_negative_prompt:
							raise NotImplementedError
							negative_text_init = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)']*im_lq_pch.size(0)
							# negative_text_init = ['Bad photo.']*im_lq_pch.size(0)
							nega_semantic_c = model.cond_stage_model(negative_text_init)
						noise = torch.randn_like(init_latent)
						# TODO: If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
						# t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
						# t = t.to(device).long()
						# x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
						x_T = None
						# samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=im_lq_pch, batch_size=im_lq_pch.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)
						samples = model.sample(
							c, shape=init_latent.shape[1:], uc=uc, batch_size=opt.n_samples, struct_cond=z_cf, **sampling_kwargs
						)
						x_samples = model.decode_first_stage(samples)
						if ori_size is not None:
							x_samples = x_samples[:, :, :ori_size[-2], :ori_size[-1]]
						if opt.colorfix_type == 'adain':
							x_samples = adaptive_instance_normalization(x_samples, init_image)
						elif opt.colorfix_type == 'wavelet':
							x_samples = wavelet_reconstruction(x_samples, init_image)
						im_sr = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
					if init_image.shape[2] > opt.vqgantile_size or init_image.shape[3] > opt.vqgantile_size:
						# adjust color, will have impact before/after ImSplitter.gather()
						if opt.colorfix_type == 'adain':
							im_sr = adaptive_instance_normalization(im_sr, init_image)
						elif opt.colorfix_type == 'wavelet':
							im_sr = wavelet_reconstruction(im_sr, init_image)
						im_sr = torch.clamp((im_sr + 1.0) / 2.0, min=0.0, max=1.0)
					if upsample_scale > opt.upscale:
							im_sr = F.interpolate(
										im_sr,
										size=(int(init_image.size(-2)*opt.upscale/upsample_scale),
											  int(init_image.size(-1)*opt.upscale/upsample_scale)),
										mode='bicubic',
										)
							im_sr = torch.clamp(im_sr, min=0.0, max=1.0)
					im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c

					if flag_pad:
						im_sr = im_sr[:, :ori_h, :ori_w, ]
					
					img_name = img_list.pop(0)

					for i in range(len(pch_out)):
						pch = pch_out[i]
						basename = os.path.splitext(os.path.basename(img_name))[0]
						pch = pch.squeeze(0).cpu().numpy().transpose(1,2,0)*255
						Image.fromarray(pch.astype(np.uint8)).save(
							os.path.join(outpath, basename+f'{i}.png'))
					for i in range(init_image.size(0)):
						# img_name = img_list.pop(0)
						basename = os.path.splitext(os.path.basename(img_name))[0]
						# x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(im_sr[i, ].astype(np.uint8)).save(
							os.path.join(outpath, basename+'.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()
