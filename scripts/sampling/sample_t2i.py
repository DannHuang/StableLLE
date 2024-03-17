import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import argparse
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from scripts.demo.streamlit_helpers import init_embedder_options, init_sampling, do_sample, perform_save_locally, load_model_from_config
from scripts.demo.sampling import VERSION2SPECS, SD_XL_BASE_RATIOS
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        const=True,
        default="sdxl",
        nargs="?",
        help="model arch",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        help="output image path"
    )
    parser.add_argument(
        "--return_latents",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="return latents",
    )
    parser.add_argument(
        "--load_filter",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="filter sample",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        help="prompt"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="negative prompt"
    )
    # parser.add_argument(
    #     "-s",
    #     "--seed",
    #     type=int,
    #     default=23,
    #     help="seed for seed_everything",
    # )
    return parser

def load_img(display=True, key=None, device="cuda"):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)

def sample(
    outpath: str = "assets/test_image.png",  # Can either be image file or folder with image files
    num_steps: Optional[int] = None,
    version: str = "sdxl",
    seed: int = 42,
    device: str = "cuda",
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    version_dict = VERSION2SPECS[version]

    model, filter = load_model(
        model_config,
        device,
        num_steps,
    )
    torch.manual_seed(seed)

    # Additional input images

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    fps_id + 1,
                    (samples.shape[-1], samples.shape[-2]),
                )

                samples = embed_watermark(samples)
                samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                for frame in vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
                writer.release()

def txt2img(opt):
    version_dict = VERSION2SPECS[opt.version]
    if opt.version.startswith("SDXL"):
        W, H = SD_XL_BASE_RATIOS['1.0'] # TODO: add options here
    else:
        H, W = version_dict['H'], version_dict['W']
    C = version_dict["C"]
    F = version_dict["f"]
    config = version_dict["config"]
    ckpt = version_dict["ckpt"]
    config = OmegaConf.load(config)
    model, msg = load_model_from_config(config, ckpt if os.path.isfile(ckpt) else None)

    if opt.load_filter:
        filter = DeepFloydDataFiltering(verbose=False)
    else:
        filter = None
    if msg: print(msg)

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }

    is_legacy = version_dict["is_legacy"]

    if not is_legacy: negative_prompt = ""  # which is unused

    stage2strength = None
    finish_denoising = False
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=opt.prompt,
        negative_prompt=opt.negative_prompt,
    )
    sampler, num_rows, num_cols = init_sampling(stage2strength=stage2strength)
    num_samples = num_rows * num_cols

    out = do_sample(
        model,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=opt.return_latents,
        filter=filter,
    )
    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None

    # if add_pipeline and samples_z is not None:
    #     st.write("**Running Refinement Stage**")
    #     samples = apply_refiner(
    #         samples_z,
    #         state2,
    #         sampler2,
    #         samples_z.shape[0],
    #         prompt=prompt,
    #         negative_prompt=negative_prompt if is_legacy else "",
    #         filter=state.get("filter"),
    #         finish_denoising=finish_denoising,
    #     )

    if opt.outpath is not None and samples is not None:
        perform_save_locally(opt.outpath, samples)

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    # config.model.params.sampler_config.params.guider_config.params.num_frames = (
    #     num_frames
    # )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()
    txt2img(opt)
