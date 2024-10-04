from venv import logger
import torch.nn as nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for _, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def group_weight(module, norm_layer, lr):
    if norm_layer == "BatchNorm2d":
        norm_layer = nn.BatchNorm2d
    elif norm_layer == "SyncBN":
        norm_layer == nn.SyncBatchNorm
    else:
        logger.error("Unsupported norm layer")
    weight_group = []
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(
            m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif (
            isinstance(m, norm_layer)
            or isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
            or isinstance(m, nn.GroupNorm)
            or isinstance(m, nn.LayerNorm)
        ):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
    return weight_group
