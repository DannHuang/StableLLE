import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )

class IrWrapper(nn.Module):
    def __init__(self, diffusion_model, struct_cond_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)
        self.struct_cond_model = compile(struct_cond_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class IrDiffusionWrapper(IrWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, struct_cond: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        struct_cond_dict = self.struct_cond_model(struct_cond, t)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            struct_cond=struct_cond_dict,
            y=c.get("vector", None),
            **kwargs,
        )
