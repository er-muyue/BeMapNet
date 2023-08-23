import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from torch.utils.checkpoint import checkpoint
from bemapnet.models.utils.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from bemapnet.models.utils.position_encoding import PositionEmbeddingIPM, PositionEmbeddingTgt


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels,
        src_shape=(32, 336),
        query_shape=(32, 32),
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        src_pos_embed='sine',
        tgt_pos_embed='sine',
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        use_checkpoint=True,
        ipm_proj_conf=None,
        ipmpe_with_sine=True,
        enforce_no_aligner=False,
    ):
        super().__init__()

        self.src_shape = src_shape
        self.query_shape = query_shape
        self.d_model = d_model
        self.nhead = nhead
        self.ipm_proj_conf = ipm_proj_conf

        self.ipmpe_with_sine = ipmpe_with_sine
        self.enforce_no_aligner = enforce_no_aligner

        num_queries = np.prod(query_shape).item()
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, d_model)
        src_pe, tgt_pe = self._get_pos_embed_layers(src_pos_embed, tgt_pos_embed)
        self.src_pos_embed_layer, self.tgt_pos_embed_layer = src_pe, tgt_pe

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, use_checkpoint)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate_dec, use_checkpoint
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_pos_embed_layers(self, src_pos_embed, tgt_pos_embed):

        pos_embed_encoder = None
        if (src_pos_embed.startswith('ipm_learned')) and (tgt_pos_embed.startswith('ipm_learned')) \
                and not self.enforce_no_aligner:
            pos_embed_encoder = nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.d_model * 4, self.d_model, kernel_size=1, stride=1, padding=0)
            )

        if src_pos_embed == 'sine':
            src_pos_embed_layer = PositionEmbeddingSine(self.d_model // 2, normalize=True)
        elif src_pos_embed == 'learned':
            src_pos_embed_layer = PositionEmbeddingLearned(self.src_shape, self.d_model)
        elif src_pos_embed == 'ipm_learned':
            input_shape = self.ipm_proj_conf['input_shape']
            src_pos_embed_layer = PositionEmbeddingIPM(
                pos_embed_encoder, self.src_shape, input_shape, num_pos_feats=self.d_model,
                sine_encoding=self.ipmpe_with_sine)
        else:
            raise NotImplementedError
        self.src_pos_embed = src_pos_embed

        if tgt_pos_embed == 'sine':
            tgt_pos_embed_layer = PositionEmbeddingSine(self.d_model // 2, normalize=True)
        elif tgt_pos_embed == 'learned':
            tgt_pos_embed_layer = PositionEmbeddingLearned(self.query_shape, self.d_model)
        elif tgt_pos_embed == 'ipm_learned':
            map_size, map_res = self.ipm_proj_conf['map_size'], self.ipm_proj_conf['map_resolution']
            tgt_pos_embed_layer = PositionEmbeddingTgt(
                pos_embed_encoder, self.query_shape, map_size, map_res, num_pos_feats=self.d_model, sine_encoding=True)
        else:
            raise NotImplementedError
        self.tgt_pos_embed = tgt_pos_embed

        return src_pos_embed_layer, tgt_pos_embed_layer

    def forward(self, src, mask=None, cameras_info=None):

        bs, c, h, w = src.shape
        if mask is None:
            mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)  # (B, H, W)

        src = self.input_proj(src)  # (B, C, H, W)
        src = src.flatten(2).permute(2, 0, 1)  # (H* W, B, C)

        if self.src_pos_embed.startswith('ipm_learned'):
            extrinsic = cameras_info['extrinsic'].float()
            intrinsic = cameras_info['intrinsic'].float()
            ida_mats = cameras_info['ida_mats'].float()
            do_flip = cameras_info['do_flip']
            src_pos_embed, src_mask = self.src_pos_embed_layer(extrinsic, intrinsic, ida_mats, do_flip)
            mask = ~src_mask
        else:
            src_pos_embed = self.src_pos_embed_layer(mask)

        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1)  # (H* W, B, C)
        mask = mask.flatten(1)  # (B, H * W)

        query_embed = self.query_embed.weight  # (H* W, C)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (H* W, B, C)
        tgt = query_embed  # (H* W, B, C)

        query_mask = torch.zeros((bs, *self.query_shape), dtype=torch.bool, device=src.device)
        query_pos_embed = self.tgt_pos_embed_layer(query_mask)  # (B, C, H, W)
        query_pos_embed = query_pos_embed.flatten(2).permute(2, 0, 1)  # (H* W, B, C)

        memory = self.encoder.forward(src, None, mask, src_pos_embed)  # (H* W, B, C)
        hs = self.decoder.forward(tgt, memory, None, None, None, mask, src_pos_embed, query_pos_embed)  # (M, H* W, B, C)
        ys = hs.permute(0, 2, 3, 1)  # (M, B, C, H* W)
        ys = ys.reshape(*ys.shape[:-1], *self.query_shape)  # (M, B, C, H, W)

        return memory, hs, ys


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, use_checkpoint=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                output = checkpoint(layer, output, mask, src_key_padding_mask, pos)
            else:
                output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, use_checkpoint=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                output = checkpoint(
                    layer,
                    output,
                    memory,
                    tgt_mask,
                    memory_mask,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    pos,
                    query_pos,
                )
            else:
                output = layer(
                    output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
                )

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
