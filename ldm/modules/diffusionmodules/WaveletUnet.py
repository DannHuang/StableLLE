from ldm.modules.diffusionmodules.openaimodel import UNetModel,Downsample,Upsample
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
import torch
import torch as th
from torch import nn
from ldm.modules.attention import BasicTransformerBlock
from einops import rearrange, repeat

import pywt

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class WaveletTransformer(nn.Module):
  
    def __init__(self, 
                 in_channels, 
                 n_heads, 
                 d_head,
                 depth=1, 
                 dropout=0., 
                 context_dim=None,
                 disable_self_attn=False, 
                 use_linear=False,
                 use_checkpoint=True,
                 wavelet_name='Haar',
                 **kwargs
                 ):
        super().__init__()
        inner_dim = n_heads * d_head
        
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        
        self.cross_attn=BasicTransformerBlock(
            dim=inner_dim, n_heads=n_heads, d_head=d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False
        )
        self.self_attn=BasicTransformerBlock(
            dim=inner_dim, n_heads=n_heads, d_head=d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False
        )
        
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        
        self.norm = Normalize(in_channels)
        self.use_linear = use_linear
        self.wavelet_name=wavelet_name
        
    def forward(self, x, context=None,LL=None,return_ori_LL=False):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        LL_in=LL
        
        LL=self.norm(LL)
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
            LL=self.proj_in(LL)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        LL = rearrange(LL, 'b c h w -> b (h w) c').contiguous()

        if self.use_linear:
            x = self.proj_in(x)
            LL=self.proj_in(LL)

        x=self.cross_attn(x,context=LL)
        LL=self.self_attn(LL)
       
        if self.use_linear:
            x = self.proj_out(x)
            LL = self.proj_out(LL)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        LL = rearrange(LL, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        if not self.use_linear:
            x = self.proj_out(x)
            LL = self.proj_out(LL)
        
        LL=LL+LL_in
        if return_ori_LL:
            return x+x_in,LL
        
        LA, (LH, LV, LD)=pywt.dwt2(LL.detach().cpu().numpy(), self.wavelet_name)
        # LL_in_A,(LL_in_H,LL_in_V,LL_in_D)=pywt.dwt2(LL_in.detach().cpu().numpy(), self.wavelet_name)
        LL=torch.tensor([LA,LH,LV,LD]).to(x.device)
        _,B,C,H,W=LL.shape
        LL=LL.reshape(B,C*4,H,W)
        
        return x + x_in,LL


class WaveletUnet(UNetModel):
    def __init__(
        self,
        wavelet_name='Haar',
       **kwargs
    ):
        super().__init__(**kwargs)
        self.wave_trans_block=nn.Sequential(
            WaveletTransformer(
            in_channels=kwargs['model_channels'],
            n_heads=5,
            d_head=64,
            use_linear=True,
            wavelet_name='Haar' 
        ),
            WaveletTransformer(
            in_channels=kwargs['model_channels']*2,
            n_heads=5,
            d_head=64*2,
            use_linear=True,
            wavelet_name='Haar' 
        ),
            WaveletTransformer(
            in_channels=kwargs['model_channels']*4,
            n_heads=5,
            d_head=64*4,
            use_linear=True,
            wavelet_name='Haar' 
        ),
            WaveletTransformer(
            in_channels=kwargs['model_channels']*4,
            n_heads=5,
            d_head=64*4,
            use_linear=True,
            wavelet_name='Haar' 
        ),
            WaveletTransformer(
            in_channels=kwargs['model_channels']*4,
            n_heads=5,
            d_head=64*4,
            use_linear=True,
            wavelet_name='Haar' 
        ),
            WaveletTransformer(
            in_channels=kwargs['model_channels']*2,
            n_heads=5,
            d_head=64*2,
            use_linear=True,
            wavelet_name='Haar' 
        )
            )
        
        self.channel_reducetion_linear=nn.Sequential(
            #downsample
            nn.Conv2d(kwargs['model_channels']*4,kwargs['model_channels']*2,kernel_size=1),
            nn.Conv2d(kwargs['model_channels']*8,kwargs['model_channels']*4,kernel_size=1),
            nn.Conv2d(kwargs['model_channels']*16,kwargs['model_channels']*8,kernel_size=1),
            #upsample
            nn.Conv2d(kwargs['model_channels']*8,kwargs['model_channels']*16,kernel_size=1),
            nn.Conv2d(kwargs['model_channels']*4,kwargs['model_channels']*16,kernel_size=1),
            nn.Conv2d(kwargs['model_channels']*4,kwargs['model_channels']*8,kernel_size=1),

        )
        
        self.low_light_proj_in=nn.Conv2d(in_channels=4,out_channels=320,kernel_size=1)
        self.wavelet_name=wavelet_name
        
    def forward(self, 
                x,
                timesteps=None,
                context=None,
                y=None,
                low_light_feat=None,
                **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        wave_block_idx=0
        
        LL=low_light_feat
        low_light_feat=self.low_light_proj_in(low_light_feat)
        for module in self.input_blocks:   
            if type(module[0]) == Downsample:
                h,low_light_feat=self.wave_trans_block[wave_block_idx](h,LL=low_light_feat)
                low_light_feat=self.channel_reducetion_linear[wave_block_idx](low_light_feat)
                wave_block_idx+=1
            h = module(h, emb, context)
            hs.append(h)
            
        h = self.middle_block(h, emb, context)
        
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if type(module[-1])==Upsample:
                low_light_feat=self.channel_reducetion_linear[wave_block_idx](low_light_feat)
                B,C,H,W=low_light_feat.shape
                low_light_feat=low_light_feat.reshape(4,B,C//4,H,W)
                LA,LH,LV,LD=low_light_feat[0].detach().cpu().numpy(),low_light_feat[1].detach().cpu().numpy(),low_light_feat[2].detach().cpu().numpy(),low_light_feat[3].detach().cpu().numpy()
                low_light_feat = torch.tensor(pywt.idwt2((LA, (LH, LV, LD)), self.wavelet_name)).to(h.device)
                h,low_light_feat=self.wave_trans_block[wave_block_idx](h,LL=low_light_feat,return_ori_LL=True)
                wave_block_idx+=1
                
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


# x=torch.rand(2,320,64,64)
# LL=torch.rand(2,320,64,64)

# x_out,LL_out=trans_block(x,context=None,LL=LL)

# print('debug')