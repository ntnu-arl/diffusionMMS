
from models.backbone.dual_dat import dual_dat_b, dual_dat_s, dual_dat_t
from models.decoder.fcn import FCNHead
from models.decoder.deformable_transformer import MaskDecoder
from models.decoder.neck import FeaturePyramidNetwork, MultiStageMerging

AVAI_BACKBONE = {
    "dat_t": dual_dat_t,
    "dat_s": dual_dat_s,
    "dat_b": dual_dat_b,
}
#
AVAI_DECODER = {
    "DeformableDETRwithTime": MaskDecoder,
    "fcn": FCNHead,
}

AVAI_NECK = {
    "fpn": FeaturePyramidNetwork,
    "MultiStageMerging": MultiStageMerging,
}


def get_decoder(model_name, **kwargs):
    if model_name not in AVAI_DECODER:
        print("not supported decoder name, please implement it first.")
    return AVAI_DECODER[model_name](**kwargs)


def get_backbone(model_name, **kwargs):
    if model_name not in AVAI_BACKBONE:
        print("not supported backbone name, please implement it first.")
    return AVAI_BACKBONE[model_name](**kwargs)


def get_neck(type_name, **kwargs):
    if type_name not in AVAI_NECK:
        print("not supported positional encoding name, please implement it first.")
    return AVAI_NECK[type_name](**kwargs)
