from .ddpm import LatentDiffusion
from torch import nn
import torch
from ldm.util import instantiate_from_config
from .ddim import DDIMSampler


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)#illumination prior map
        # stx()
        input_ = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input_)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class LLIEdiffusion(nn.Module):
    def __init__(self,
                 diffusion_config=None,
                 diffusion_ckpt=None,
                 estimator_middle_channels=40,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.estimator=Illumination_Estimator(
            n_fea_middle=estimator_middle_channels, n_fea_in=4, n_fea_out=3
        )
        self.latent_diffusion=self.load_model_from_config(
            config=diffusion_config,
            ckpt=diffusion_ckpt
        )       
        self.sampler=DDIMSampler(self.latent_diffusion)
         
        
    def forward(
        self,
        **kwargs
    ):
        print('--------debug--------')

    
    def load_model_from_config(self,config, ckpt, verbose=False):
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

        model.cuda()
        # model.eval()
        return model