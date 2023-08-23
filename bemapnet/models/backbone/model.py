import torch
import torch.nn as nn
import torch.nn.functional as F
from bemapnet.models.backbone.resnet import ResNet
from bemapnet.models.backbone.efficientnet import EfficientNet
from bemapnet.models.backbone.swin_transformer import SwinTransformer
from bemapnet.models.backbone.bifpn import BiFPN


class ResNetBackbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(ResNetBackbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb = ResNet(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape
        self.bkb.init_weights()

    def forward(self, inputs):
        images = inputs["images"]
        images = images.view(-1, *images.shape[-3:])
        bkb_features = list(self.bkb(images)[-self.ret_layers:])
        nek_features = self.fpn(bkb_features) if self.fpn is not None else None
        return {"im_bkb_features": bkb_features, "im_nek_features": nek_features}


class EfficientNetBackbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(EfficientNetBackbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb = EfficientNet.from_pretrained(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape
        del self.bkb._conv_head
        del self.bkb._bn1
        del self.bkb._avg_pooling
        del self.bkb._dropout
        del self.bkb._fc

    def forward(self, inputs):
        images = inputs["images"]
        images = images.view(-1, *images.shape[-3:])
        endpoints = self.bkb.extract_endpoints(images)
        bkb_features = []
        for i, (key, value) in enumerate(endpoints.items()):
            if i > 0:
                bkb_features.append(value)
        bkb_features = list(bkb_features[-self.ret_layers:])
        nek_features = self.fpn(bkb_features) if self.fpn is not None else None
        return {"im_bkb_features": bkb_features, "im_nek_features": nek_features}


class SwinTRBackbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(SwinTRBackbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb = SwinTransformer(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape

    def forward(self, inputs):
        images = inputs["images"]
        images = images.view(-1, *images.shape[-3:])
        bkb_features = list(self.bkb(images)[-self.ret_layers:])
        nek_features = None
        if self.fpn is not None:
            nek_features = self.fpn(bkb_features)
        else:
            if self.up_shape is not None:
                nek_features = [torch.cat([self.up_sample(x, self.up_shape) for x in bkb_features], dim=1)]

        return {"im_bkb_features": bkb_features, "im_nek_features": nek_features}

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)
