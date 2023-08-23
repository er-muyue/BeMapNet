import torch.nn as nn
from bemapnet.models import backbone, bev_decoder, ins_decoder, output_head
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
# warnings.filterwarnings('ignore')


class BeMapNet(nn.Module):
    def __init__(self, model_config, *args, **kwargs):
        super(BeMapNet, self).__init__()
        self.im_backbone = self.create_backbone(**model_config["im_backbone"])
        self.bev_decoder = self.create_bev_decoder(**model_config["bev_decoder"])
        self.ins_decoder = self.create_ins_decoder(**model_config["ins_decoder"])
        self.output_head = self.create_output_head(**model_config["output_head"])
        self.post_processor = self.create_post_processor(**model_config["post_processor"])

    def forward(self, inputs):
        outputs = {}
        outputs.update({k: inputs[k] for k in ["images", "extra_infos"]})
        outputs.update({k: inputs[k].float()for k in ["extrinsic", "intrinsic", "ida_mats"]})
        outputs.update(self.im_backbone(outputs))
        outputs.update(self.bev_decoder(outputs))
        outputs.update(self.ins_decoder(outputs))
        outputs.update(self.output_head(outputs))
        return outputs

    @staticmethod
    def create_backbone(arch_name, ret_layers, bkb_kwargs, fpn_kwargs, up_shape=None):
        __factory_dict__ = {
            "resnet": backbone.ResNetBackbone,
            "efficient_net": backbone.EfficientNetBackbone,
            "swin_transformer": backbone.SwinTRBackbone,
        }
        return __factory_dict__[arch_name](bkb_kwargs, fpn_kwargs, up_shape, ret_layers)

    @staticmethod
    def create_bev_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "transformer": bev_decoder.TransformerBEVDecoder,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_ins_decoder(arch_name, net_kwargs):
        __factory_dict__ = {
            "mask2former": ins_decoder.Mask2formerINSDecoder,
        }

        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_output_head(arch_name, net_kwargs):
        __factory_dict__ = {
            "bezier_output_head": output_head.PiecewiseBezierMapOutputHead,
        }
        return __factory_dict__[arch_name](**net_kwargs)

    @staticmethod
    def create_post_processor(arch_name, net_kwargs):
        __factory_dict__ = {
            "bezier_post_processor": output_head.PiecewiseBezierMapPostProcessor,
        }
        return __factory_dict__[arch_name](**net_kwargs)
