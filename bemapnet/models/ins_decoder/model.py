import torch.nn as nn
import torch.nn.functional as F
from bemapnet.models.ins_decoder.mask2former import MultiScaleMaskedTransformerDecoder
import time


class Mask2formerINSDecoder(nn.Module):
    def __init__(self, decoder_ids=(5, ), tgt_shape=None, **kwargs):
        super(Mask2formerINSDecoder, self).__init__()
        self.decoder_ids = tuple(decoder_ids)  # [0, 1, 2, 3, 4, 5]
        self.tgt_shape = tgt_shape
        self.bev_decoder = MultiScaleMaskedTransformerDecoder(**kwargs)

    def forward(self, inputs):
        assert "bev_enc_features" in inputs
        bev_enc_features = inputs["bev_enc_features"]
        if self.tgt_shape is not None:
            bev_enc_features = [self.up_sample(x) for x in inputs["bev_enc_features"]]
        out = self.bev_decoder(bev_enc_features[-1:], bev_enc_features[-1])
        return {"mask_features": [out["pred_masks"][1:][i] for i in self.decoder_ids],
                "obj_scores": [out["pred_logits"][1:][i] for i in self.decoder_ids],
                "decoder_outputs": [out["decoder_outputs"][1:][i] for i in self.decoder_ids],
                "bev_enc_features": bev_enc_features}

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)
