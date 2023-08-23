"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos=(50, 50), num_pos_feats=256):
        super().__init__()
        self.num_pos = num_pos
        self.pos_embed = nn.Embedding(num_pos[0] * num_pos[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_embed.weight)

    def forward(self, mask):
        h, w = mask.shape[-2:]
        pos = self.pos_embed.weight.view(*self.num_pos, -1)[:h, :w]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingIPM(nn.Module):

    def __init__(self,
                 encoder=None,
                 num_pos=(16, 168),
                 input_shape=(512, 896),
                 num_pos_feats=64,
                 sine_encoding=False,
                 temperature=10000):
        super().__init__()

        h, w_expand = num_pos
        self.current_shape = (h, w_expand // 6)
        self.input_shape = input_shape

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.encoder = encoder
        self.sine_encoding = sine_encoding

    def get_embedding(self, extrinsic, intrinsic, ida_mats):
        """
        Get the BeV Coordinate for Image

        Return
            xy_world_coord (N, H, W, 2) Ego x, y coordinate
            Valid (N, H, W, 1) -- Valid Points or Not 1 -- valid; 0 -- invalid
        """
        # extrinsic -> (B, M, 4, 4)
        device, b, n = extrinsic.device, extrinsic.shape[0], extrinsic.shape[1]

        x = torch.linspace(0, self.input_shape[1] - 1, self.current_shape[1], dtype=torch.float)
        y = torch.linspace(0, self.input_shape[0] - 1, self.current_shape[0], dtype=torch.float)
        y_grid, x_grid = torch.meshgrid(y, x)
        z = torch.ones(self.current_shape)
        feat_coords = torch.stack([x_grid, y_grid, z], dim=-1).to(device)  # (H, W, 3)
        feat_coords = feat_coords.unsqueeze(0).repeat(n, 1, 1, 1).unsqueeze(0).repeat(b, 1, 1, 1, 1)  # (B, N, H, W, 3)

        ida_mats = ida_mats.view(b, n, 1, 1, 3, 3)
        image_coords = ida_mats.inverse().matmul(feat_coords.unsqueeze(-1))  # (B, N, H, W, 3, 1)

        intrinsic = intrinsic.view(b, n, 1, 1, 3, 3)  # (B, N, 1, 1, 3, 3)
        normed_coords = torch.linalg.inv(intrinsic) @ image_coords  # (B, N, H, W, 3, 1)

        ext_rots = extrinsic[:, :, :3, :3]  # (B, N, 3, 3)
        ext_trans = extrinsic[:, :, :3, 3]  # (B, N, 3)

        ext_rots = ext_rots.view(b, n, 1, 1, 3, 3)  # (B, N, 1, 1, 3, 3)
        world_coords = (ext_rots @ normed_coords).squeeze(-1)  # (B, N, H, W, 3)
        world_coords = F.normalize(world_coords, p=2, dim=-1)
        z_coord = world_coords[:, :, :, :, 2]  # (B, N, H, W)

        trans_z = ext_trans[:, :, 2].unsqueeze(-1).unsqueeze(-1)   # (B, N, 1, 1)
        depth = - trans_z / z_coord  # (B, N, H, W)
        valid = depth > 0  # (B, N, H, W)

        xy_world_coords = world_coords[:, :, :, :, :2]  # (B, N, H, W, 2)
        xy_world_coords = xy_world_coords * depth.unsqueeze(-1)
        valid = valid.unsqueeze(-1)  # (B, N, H, W, 1)

        return xy_world_coords, valid

    def forward(self, extrinsic, intrinsic, ida_mats, do_flip):
        """
        extrinsic (N, 6, 4, 4) torch.Tensor
        intrinsic (N, 6, 3, 3)
        """
        device = extrinsic.device
        xy_pos_embed, valid = self.get_embedding(extrinsic, intrinsic, ida_mats)
        if do_flip:
            xy_pos_embed[:, :, :, :, 1] = -1 * xy_pos_embed[:, :, :, :, 1]
        # along with w
        xy_pos_embed = torch.cat(torch.unbind(xy_pos_embed, dim=1), dim=-2)  # (B, H, N*W, 2)
        valid = torch.cat(torch.unbind(valid, dim=1), dim=-2)  # (B, H, N*W, 2)
        if self.sine_encoding:
            # Use Sine encoding to get 256 dim embeddings
            dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_pos_feats // 2))
            pos_embed = xy_pos_embed[:, :, :, :, None] / dim_t
            pos_x = torch.stack((pos_embed[:, :, :, 0, 0::2].sin(), pos_embed[:, :, :, 0, 1::2].cos()), dim=4)
            pos_y = torch.stack((pos_embed[:, :, :, 1, 0::2].sin(), pos_embed[:, :, :, 1, 1::2].cos()), dim=4)
            pos_full_embed = torch.cat((pos_y.flatten(3), pos_x.flatten(3)), dim=3)
            pos_combined = torch.where(valid, pos_full_embed, torch.tensor(0., dtype=torch.float32, device=device))
            pos_combined = pos_combined.permute(0, 3, 1, 2)  # (B, 2, H, W')
        else:
            assert None
            # pos_combined = torch.where(valid, xy_pos_embed, torch.tensor(0., dtype=torch.float32, device=device))
            # pos_combined = pos_combined.permute(0, 3, 1, 2)

        if self.encoder is None:
            return pos_combined, valid.squeeze(-1)
        else:
            pos_embed_contiguous = pos_combined.contiguous()
            return self.encoder(pos_embed_contiguous), valid.squeeze(-1)


class PositionEmbeddingTgt(nn.Module):
    def __init__(self,
                 encoder=None,
                 tgt_shape=(40, 20),
                 map_size=(400, 200),
                 map_resolution=0.15,
                 num_pos_feats=64,
                 sine_encoding=False,
                 temperature=10000):
        super().__init__()
        self.tgt_shape = tgt_shape
        self.encoder = encoder
        self.map_size = map_size
        self.map_resolution = map_resolution
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.sine_encoding = sine_encoding

    def forward(self, mask):
        B = mask.shape[0]

        map_forward_ratio = self.tgt_shape[0] / self.map_size[0]
        map_lateral_ratio = self.tgt_shape[1] / self.map_size[1]

        map_forward_res = self.map_resolution / map_forward_ratio
        map_lateral_res = self.map_resolution / map_lateral_ratio

        X = (torch.arange(self.tgt_shape[0] - 1, -1, -1, device=mask.device) + 0.5 - self.tgt_shape[
            0] / 2) * map_forward_res
        Y = (torch.arange(self.tgt_shape[1] - 1, -1, -1, device=mask.device) + 0.5 - self.tgt_shape[
            1] / 2) * map_lateral_res

        grid_X, grid_Y = torch.meshgrid(X, Y)
        pos_embed = torch.stack([grid_X, grid_Y], dim=-1)  # (H, W, 2)

        if self.sine_encoding:
            dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=mask.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_pos_feats // 2))

            pos_embed = pos_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_embed[:, :, 0, 0::2].sin(), pos_embed[:, :, 0, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_embed[:, :, 1, 0::2].sin(), pos_embed[:, :, 1, 1::2].cos()), dim=3).flatten(2)
            pos_full_embed = torch.cat((pos_y, pos_x), dim=2)

            pos_embed = pos_full_embed.unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
        else:
            pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)

        if self.encoder is None:
            return pos_embed
        else:
            pos_embed_contiguous = pos_embed.contiguous()
            return self.encoder(pos_embed_contiguous)