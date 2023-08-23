import os
import torch
from .model import SwinTransformer as _SwinTransformer
from torch.utils import model_zoo

model_urls = {
    "tiny": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth",
    "base": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth",
}


class SwinTransformer(_SwinTransformer):
    def __init__(
        self,
        arch="tiny",
        pretrained=False,
        window_size=7,
        shift_mode=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        **kwargs
    ):
        if arch == "tiny":
            embed_dim = 96
            depths = (2, 2, 6, 2)
            num_heads = (3, 6, 12, 24)
        elif arch == "small":
            embed_dim = 96
            depths = (2, 2, 18, 2)
            num_heads = (3, 6, 12, 24)
        elif arch == "base":
            embed_dim = 128
            depths = (2, 2, 18, 2)
            num_heads = (4, 8, 16, 32)
        else:
            raise NotImplementedError

        super(SwinTransformer, self).__init__(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            shift_mode=shift_mode,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            out_indices=out_indices,
            use_checkpoint=use_checkpoint,
            **kwargs
        )
        if isinstance(pretrained, bool):
            assert pretrained is True
            print(model_urls[arch])
            state_dict = model_zoo.load_url(model_urls[arch])["state_dict"]
        elif isinstance(pretrained, str):
            assert os.path.exists(pretrained)
            print(pretrained)
            state_dict = torch.load(pretrained)["state_dict"]
        else:
            raise NotImplementedError

        self.arch = arch
        self.init_weights(state_dict=state_dict)

    def init_weights(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "backbone" in key:
                new_state_dict[key.replace("backbone.", "")] = value
        ret = self.load_state_dict(new_state_dict, strict=False)
        print("Backbone missing_keys: {}".format(ret.missing_keys))
        print("Backbone unexpected_keys: {}".format(ret.unexpected_keys))
