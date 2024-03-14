import torch.nn as nn
import pytorch_lightning as pl
from ...util import instantiate_from_config

class StructEncoderWT(pl.LightningModule):
    def __init__(self, configs, encoder_config, used_levels):
        super().__init__()
        self.used_levels = used_levels
        self.encoders = nn.ModuleDict()
        for res, param_dict in configs.items():
            if res not in self.used_levels: continue
            encoder_config.params.in_channels = param_dict['in_channels']
            encoder_config.params.attention_resolutions = param_dict['attn_reso']
            self.encoders[res] = instantiate_from_config(encoder_config)


    def forward(self, struct_cond: dict, t):
        hq_encoding_dict = dict()
        for in_reso in struct_cond.keys():
            out_reso = str(int(in_reso)//8)
            hq_encoding_dict[out_reso] = self.encoders[in_reso](struct_cond[in_reso], t)
            print(f"processed {in_reso} feature: {struct_cond[in_reso].shape} -> {hq_encoding_dict[out_reso].shape}")
        return hq_encoding_dict