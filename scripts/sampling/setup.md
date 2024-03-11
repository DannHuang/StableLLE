# Stable LLIE

## Change Log
**March 10, 2024**
- Introduction of setting up SDXL locally.

## Setting up

#### 1. Clone the repo

```shell
git clone https://github.com/Stability-AI/generative-models.git
cd generative-models
```

#### 2. Create virtualenv

**NOTE:** Assuming in the project root.

**Python 3.10 + PyTorch 2.0**

```shell
# install required packages from pypi
conda create -n sd python=3.10
conda activate sd
pip3 install -r requirements/pt2.txt
```

#### 3. Install `sgm`

```shell
pip3 install .
```

#### 4. Install `sdata` for training
Datapipline created by stabilityAI, first clone the repository and change working directory.

```shell
git clone https://github.com/Stability-AI/datapipelines.git
cd /your/path/to/datapiplines
pip3 install -e .
```

## Download models
To avoid network issues, download following files to local directory:

- **Diffusion Models**
  - SDXL: (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/)
    - sd_xl_base_1.0.safetensors
  - SD-2.1: (https://huggingface.co/stabilityai/stable-diffusion-2-1-base/)
    - v2-1_512-ema-pruned.ckpt
<!-- - [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) -->
<!-- - [SD-2.1-768](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors) -->
- **Text Encoders**
  - ViT-L-14: (https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
    - pytorch_model.bin
    - config.json
    - merges.txt
    - tokenizer_config.json
    - special_tokens_map.json
    - vocab.json
  
    Save above files to same directory.
  - ViT-bigG-14: (https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main)
    - open_clip_pytorch_model.bin

## Configuration
Modify configs to specify downloaded files.
  - configs/inference/sd_xl_base.yaml
    - line 6: ```ckpt_path: {/your/path/to/sd_xl_base_1.0.safetensors}```
    - line 46: ```local_dir: {/your/path/to/ViT-L-14/files}```
    - line 54: ```version: {/your/path/to/ViT-bigG-14/open_clip_pytorch_model.bin}```

## Training:

We are providing example training configs in `configs/example_training`. To launch a training, run

```
python main.py --base configs/<config1.yaml> configs/<config2.yaml>
```

where configs are merged from left to right (later configs overwrite the same values).
This can be used to combine model, training and data configs. However, all of them can also be
defined in a single config. For example, to run a class-conditional pixel-based diffusion model training on MNIST,
run

```bash
python main.py --base configs/example_training/toy/mnist_cond.yaml
```

**NOTE 1:** Using the non-toy-dataset
configs `configs/example_training/imagenet-f8_cond.yaml`, `configs/example_training/txt2img-clipl.yaml`
and `configs/example_training/txt2img-clipl-legacy-ucg-training.yaml` for training will require edits depending on the
used dataset (which is expected to stored in tar-file in
the [webdataset-format](https://github.com/webdataset/webdataset)). To find the parts which have to be adapted, search
for comments containing `USER:` in the respective config.

**NOTE 2:** This repository supports both `pytorch1.13` and `pytorch2`for training generative models. However for
autoencoder training as e.g. in `configs/example_training/autoencoder/kl-f4/imagenet-attnfree-logvar.yaml`,
only `pytorch1.13` is supported.

**NOTE 3:** Training latent generative models (as e.g. in `configs/example_training/imagenet-f8_cond.yaml`) requires
retrieving the checkpoint from [Hugging Face](https://huggingface.co/stabilityai/sdxl-vae/tree/main) and replacing
the `CKPT_PATH` placeholder in [this line](configs/example_training/imagenet-f8_cond.yaml#81). The same is to be done
for the provided text-to-image configs.

### Building New Diffusion Models

#### Conditioner

The `GeneralConditioner` is configured through the `conditioner_config`. Its only attribute is `emb_models`, a list of
different embedders (all inherited from `AbstractEmbModel`) that are used to condition the generative model.
All embedders should define whether or not they are trainable (`is_trainable`, default `False`), a classifier-free
guidance dropout rate is used (`ucg_rate`, default `0`), and an input key (`input_key`), for example, `txt` for
text-conditioning or `cls` for class-conditioning.
When computing conditionings, the embedder will get `batch[input_key]` as input.
We currently support two to four dimensional conditionings and conditionings of different embedders are concatenated
appropriately.
Note that the order of the embedders in the `conditioner_config` is important.

#### Network

The neural network is set through the `network_config`. This used to be called `unet_config`, which is not general
enough as we plan to experiment with transformer-based diffusion backbones.

#### Loss

The loss is configured through `loss_config`. For standard diffusion model training, you will have to
set `sigma_sampler_config`.

#### Sampler config

As discussed above, the sampler is independent of the model. In the `sampler_config`, we set the type of numerical
solver, number of steps, type of discretization, as well as, for example, guidance wrappers for classifier-free
guidance.

### Dataset Handling

For large scale training we recommend using the data pipelines from
our [data pipelines](https://github.com/Stability-AI/datapipelines) project. The project is contained in the requirement
and automatically included when following the steps from the [Installation section](#installation).
Small map-style datasets should be defined here in the repository (e.g., MNIST, CIFAR-10, ...), and return a dict of
data keys/values,
e.g.,

```python
example = {"jpg": x,  # this is a tensor -1...1 chw
           "txt": "a beautiful image"}
```

where we expect images in -1...1, channel-first format.
