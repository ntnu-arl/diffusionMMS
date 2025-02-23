import torch
from ..models.segmentor.diffusionmms import DiffusionMMS

AVAI_MODEL = {"diffusionmms": DiffusionMMS}
AVAI_OPT = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

def get_model(model_name, **kwargs):
    if model_name not in AVAI_MODEL:
        print("not supported model name, please implement it first.")
    return AVAI_MODEL[model_name](**kwargs)


def get_optimizer(opt_name, **kwargs):
    if opt_name not in AVAI_OPT:
        print("not supported optimizer name, please implement it first.")
    return AVAI_OPT[opt_name](**{k: v for k, v in kwargs.items() if v is not None})
