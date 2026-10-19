import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from functools import partial
from typing import Callable
from torch.distributions.normal import Normal

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

try:
    from mamba_ssm import Mamba
except Exception:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except Exception:
        Mamba = None


########################################################
# Networks
########################################################

def affine_para_to_theta(affine_para):
    theta = affine_para.view(-1, 3, 4)
    identity = torch.zeros_like(theta)
    identity[:, 0, 0] = 1
    identity[:, 1, 1] = 1
    identity[:, 2, 2] = 1
    return theta + identity


def affine_grid_from_para(affine_para, src_shape):
    theta = affine_para_to_theta(affine_para)
    return nnf.affine_grid(theta, src_shape, align_corners=True)


def flow_to_grid(flow):
    shape = flow.shape[2:]
    vectors = [torch.arange(0, s, device=flow.device, dtype=flow.dtype) for s in shape]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids).unsqueeze(0)

    new_locs = grid + flow
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    new_locs = new_locs.permute(0, 2, 3, 4, 1)
    return new_locs[..., [2, 1, 0]]


def grid_to_flow(grid):
    b, d, h, w, _ = grid.shape
    src_w = (grid[..., 0] / 2.0 + 0.5) * float(max(w - 1, 1))
    src_h = (grid[..., 1] / 2.0 + 0.5) * float(max(h - 1, 1))
    src_d = (grid[..., 2] / 2.0 + 0.5) * float(max(d - 1, 1))
    src = torch.stack([src_d, src_h, src_w], dim=1)

    vectors = [
        torch.arange(0, d, device=grid.device, dtype=grid.dtype),
        torch.arange(0, h, device=grid.device, dtype=grid.dtype),
        torch.arange(0, w, device=grid.device, dtype=grid.dtype),
    ]
    grids = torch.meshgrid(vectors, indexing='ij')
    identity = torch.stack(grids).unsqueeze(0)
    return src - identity


def compose_grids(grid_A, grid_def):
    """Compose affine and deformable sampling grids: grid_final = grid_A(grid_def)."""
    grid_A_channels = grid_A.permute(0, 4, 1, 2, 3)
    grid_final = nnf.grid_sample(
        grid_A_channels,
        grid_def,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return grid_final.permute(0, 2, 3, 4, 1)


class NICE_Trans(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 enc_channels: int = 8,
                 dec_channels: int = 16,
                 use_checkpoint: bool = True):
        super().__init__()

        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = Trans_decoder(in_channels=enc_channels,
                                     channel_num=dec_channels,
                                     use_checkpoint=use_checkpoint)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.AffineTransformer = AffineTransformer_block(mode='bilinear')
        self.GridTransformer = GridTransformer_block(mode='bilinear')

    def forward(self, fixed, moving):
        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        deform_flow, affine_para = self.Decoder(x_fix, x_mov)

        affine_grid = affine_grid_from_para(affine_para, moving.shape)
        deform_grid = flow_to_grid(deform_flow)
        final_grid = compose_grids(affine_grid, deform_grid)

        warped = self.GridTransformer(moving, final_grid)
        affined = self.GridTransformer(moving, affine_grid)

        return warped, deform_flow, affined, affine_para, final_grid, affine_grid, deform_grid


########################################################
# Encoder/Decoder
########################################################

class Conv_encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.conv_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.conv_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.conv_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.conv_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.conv_5 = Conv_block(channel_num * 8, channel_num * 16, use_checkpoint)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.conv_1(x_in)

        x = self.downsample(x_1)
        x_2 = self.conv_2(x)

        x = self.downsample(x_2)
        x_3 = self.conv_3(x)

        x = self.downsample(x_3)
        x_4 = self.conv_4(x)

        x = self.downsample(x_4)
        x_5 = self.conv_5(x)

        return [x_1, x_2, x_3, x_4, x_5]


class CorrGuidedContextGate(nn.Module):

    def __init__(self,
                 f_channels: int,
                 m_channels: int,
                 c_channels: int,
                 hidden_channels: int = None,
                 corr_channels: int = 27):
        super().__init__()

        if corr_channels != 27:
            raise ValueError("CorrGuidedContextGate currently supports max_disp=1 (27 correlations).")

        hidden_channels = c_channels if hidden_channels is None else hidden_channels
        spatial_hidden = max(hidden_channels // 4, 8)
        channel_hidden = max(hidden_channels // 4, 4)

        self.f_proj = nn.Conv3d(f_channels, hidden_channels, kernel_size=1)
        self.m_proj = nn.Conv3d(m_channels, hidden_channels, kernel_size=1)
        self.c_proj = nn.Conv3d(c_channels, hidden_channels, kernel_size=1)

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(corr_channels + hidden_channels, spatial_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(spatial_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_channels, channel_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out_proj = (
            nn.Identity()
            if hidden_channels == c_channels
            else nn.Conv3d(hidden_channels, c_channels, kernel_size=1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        # With the default hidden size, gamma=0 makes the whole gate an exact identity.
        if hidden_channels == c_channels:
            nn.init.dirac_(self.c_proj.weight)
            nn.init.zeros_(self.c_proj.bias)

    @staticmethod
    def _local_correlation(fixed, moving):
        _, _, d, h, w = fixed.shape
        moving = nnf.pad(moving, (1, 1, 1, 1, 1, 1))
        correlations = []

        for offset_d in (-1, 0, 1):
            for offset_h in (-1, 0, 1):
                for offset_w in (-1, 0, 1):
                    start_d = offset_d + 1
                    start_h = offset_h + 1
                    start_w = offset_w + 1
                    shifted = moving[
                        :,
                        :,
                        start_d:start_d + d,
                        start_h:start_h + h,
                        start_w:start_w + w,
                    ]
                    correlations.append((fixed * shifted).mean(dim=1, keepdim=True))

        return torch.cat(correlations, dim=1)

    def forward(self, fixed, moving, context):
        target_size = fixed.shape[2:]
        if moving.shape[2:] != target_size:
            raise ValueError(
                f"Fixed and moving feature sizes must match, got {target_size} and {moving.shape[2:]}."
            )
        if context.shape[2:] != target_size:
            context = nnf.interpolate(context, size=target_size, mode='trilinear', align_corners=True)

        fixed_p = self.f_proj(fixed)
        moving_p = self.m_proj(moving)
        context_p = self.c_proj(context)

        corr = self._local_correlation(fixed_p, moving_p)
        spatial_weight = self.spatial_attention(torch.cat([corr, context_p], dim=1))
        context_s = context_p * spatial_weight

        channel_weight = self.channel_attention(context_s)
        context_sc = context_s * channel_weight

        return self.out_proj(context_p + self.gamma * context_sc)


class Trans_decoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.trans_2 = nn.Sequential(
            MambaTrans_stage_block(embed_dim=channel_num * 2,
                                   use_checkpoint=use_checkpoint),
            MambaTrans_stage_block(embed_dim=channel_num * 2,
                                   use_checkpoint=use_checkpoint),
        )
        self.trans_3 = nn.Sequential(
            MambaTrans_stage_block(embed_dim=channel_num * 4,
                                   use_checkpoint=use_checkpoint),
            MambaTrans_stage_block(embed_dim=channel_num * 4,
                                   use_checkpoint=use_checkpoint),
        )
        self.trans_4 = nn.Sequential(
            MambaTrans_stage_block(embed_dim=channel_num * 8,
                                   use_checkpoint=use_checkpoint),
            MambaTrans_stage_block(embed_dim=channel_num * 8,
                                   use_checkpoint=use_checkpoint),
        )
        self.trans_5 = nn.Sequential(
            MambaTrans_stage_block(embed_dim=channel_num * 16,
                                   use_checkpoint=use_checkpoint),
            MambaTrans_stage_block(embed_dim=channel_num * 16,
                                   use_checkpoint=use_checkpoint),
        )

        self.backdim_2 = nn.Conv3d(in_channels * 4 + channel_num * 2, channel_num * 2, kernel_size=3, stride=1,
                                   padding='same')
        self.backdim_3 = nn.Conv3d(in_channels * 8 + channel_num * 4, channel_num * 4, kernel_size=3, stride=1,
                                   padding='same')
        self.backdim_4 = nn.Conv3d(in_channels * 16 + channel_num * 8, channel_num * 8, kernel_size=3, stride=1,
                                   padding='same')
        self.backdim_5 = nn.Conv3d(in_channels * 32, channel_num * 16, kernel_size=3, stride=1, padding='same')

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num * 8)
        self.upsample_4 = PatchExpanding_block(embed_dim=channel_num * 16)

        self.corr_gate_1 = CorrGuidedContextGate(
            f_channels=in_channels,
            m_channels=in_channels,
            c_channels=channel_num,
        )
        self.corr_gate_2 = CorrGuidedContextGate(
            f_channels=in_channels * 2,
            m_channels=in_channels * 2,
            c_channels=channel_num * 2,
        )
        self.corr_gate_3 = CorrGuidedContextGate(
            f_channels=in_channels * 4,
            m_channels=in_channels * 4,
            c_channels=channel_num * 4,
        )
        self.corr_gate_4 = CorrGuidedContextGate(
            f_channels=in_channels * 8,
            m_channels=in_channels * 8,
            c_channels=channel_num * 8,
        )

        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint)
        self.reghead_3 = DeformHead_block(channel_num * 4, use_checkpoint)
        self.reghead_4 = DeformHead_block(channel_num * 8, use_checkpoint)
        self.reghead_5 = DeformHead_block(channel_num * 16, use_checkpoint)
        self.reghead_6 = KeyMorphAffineHead(in_channels * 16, num_keypoints=16)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.GridTransformer = GridTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = x_mov

        affine_para = self.reghead_6(x_fix_5, x_mov_5, x_fix_4, x_mov_4, x_fix_3, x_mov_3)

        x_mov_1 = self._affine_warp_feature(x_mov_1, affine_para)
        x_mov_2 = self._affine_warp_feature(x_mov_2, affine_para)
        x_mov_3 = self._affine_warp_feature(x_mov_3, affine_para)
        x_mov_4 = self._affine_warp_feature(x_mov_4, affine_para)
        x_mov_5 = self._affine_warp_feature(x_mov_5, affine_para)

        # Step 1: global affine is already applied, so the deform chain starts from zero.
        x = torch.cat([x_fix_5, x_mov_5], dim=1)
        x = self.backdim_5(x)
        x_5 = self.trans_5(x)
        flow_5 = self.reghead_5(x_5)
        # 形变场，仿射矩阵

        # Step 2
        flow_5_up = self.ResizeTransformer(flow_5)
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)

        x = self.upsample_4(x_5)
        x = self.corr_gate_4(x_fix_4, x_mov_4, x)
        x = torch.cat([x_fix_4, x, x_mov_4], dim=1)
        x = self.backdim_4(x)
        x_4 = self.trans_4(x)

        x = self.reghead_4(x_4)
        flow_4 = x + flow_5_up

        # Step 3
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.upsample_3(x_4)
        x = self.corr_gate_3(x_fix_3, x_mov_3, x)
        x = torch.cat([x_fix_3, x, x_mov_3], dim=1)
        x = self.backdim_3(x)
        x_3 = self.trans_3(x)

        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up

        # Step 4
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = self.corr_gate_2(x_fix_2, x_mov_2, x)
        x = torch.cat([x_fix_2, x, x_mov_2], dim=1)
        x = self.backdim_2(x)
        x_2 = self.trans_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 5
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = self.corr_gate_1(x_fix_1, x_mov_1, x)
        x = torch.cat([x_fix_1, x, x_mov_1], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1, affine_para

    def _affine_warp_feature(self, feature, affine_para):
        affine_grid = affine_grid_from_para(affine_para, feature.shape)
        return self.GridTransformer(feature, affine_grid)


########################################################
# Blocks
########################################################

class AffineTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, affine_para):
        affine_grid = affine_grid_from_para(affine_para, src.shape)
        return nnf.grid_sample(src, affine_grid, align_corners=True, mode=self.mode)


class GridTransformer_block(nn.Module):

    def __init__(self, mode='bilinear', padding_mode='zeros'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, src, grid):
        return nnf.grid_sample(
            src,
            grid,
            align_corners=True,
            mode=self.mode,
            padding_mode=self.padding_mode,
        )


class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        new_locs = flow_to_grid(flow)
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class Conv_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm3d(out_channels)

        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm3d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in, use_reentrant=False)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out


class DeformHead_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in, use_reentrant=False)
        else:
            x_out = self.reg_head(x_in)

        return x_out


class KeyMorphAffineHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 num_keypoints: int = 16,
                 grid_size: tuple = (3, 3, 3),
                 gamma: float = 0.5,
                 sinkhorn_tau: float = 0.1,
                 sinkhorn_iters: int = 20,
                 eps: float = 1e-6):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.grid_size = grid_size
        self.gamma = gamma
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        # Keep this branch for compatibility with visualization scripts.
        self.kp_proj = nn.Conv3d(in_channels, num_keypoints, kernel_size=1, stride=1, padding=0)
        self.act = nn.Softplus()

        attn_dim = min(64, in_channels)
        num_heads = 1
        for h in [8, 4, 2, 1]:
            if attn_dim % h == 0:
                num_heads = h
                break

        self.sa_norm = nn.LayerNorm(in_channels)
        self.ca_q_norm = nn.LayerNorm(in_channels)
        self.ca_kv_norm = nn.LayerNorm(in_channels)
        self.attn_in = nn.Linear(in_channels, attn_dim, bias=False)
        self.attn_out = nn.Linear(attn_dim, in_channels, bias=False)
        self.self_attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads, batch_first=True)

        hidden_dim = max(in_channels // 2, 16)
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Lightweight descriptor projection on F4/M4 feature maps.
        f4_channels = max(in_channels // 2, 1)
        desc_channels = max(f4_channels // 2, 16)
        self.desc_proj_4 = nn.Sequential(
            nn.Conv3d(f4_channels, desc_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(desc_channels),
            nn.GELU()
        )

        offsets = torch.tensor([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0]], dtype=torch.float32)
        self.register_buffer('offsets_r1', offsets, persistent=False)
        self.register_buffer('offsets_r2', offsets * 2.0, persistent=False)
        patch_offsets = torch.tensor(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)), dtype=torch.float32)
        self.register_buffer('patch_offsets_3', patch_offsets, persistent=False)

    def forward(self, x_fix_5, x_mov_5, x_fix_4=None, x_mov_4=None, x_fix_3=None, x_mov_3=None):
        points = self.extract_correspondence_points(
            x_fix_5,
            x_mov_5,
            x_fix_4=x_fix_4,
            x_mov_4=x_mov_4,
            x_fix_3=x_fix_3,
            x_mov_3=x_mov_3,
        )

        if points['fix_points_f3'] is not None and points['mov_points_f3_refined'] is not None:
            src_points = points['fix_points_f3']
            dst_points = points['mov_points_f3_refined']
            im_shape = torch.tensor([x_fix_3.shape[4], x_fix_3.shape[3], x_fix_3.shape[2]],
                                    device=x_fix_3.device, dtype=x_fix_3.dtype)
        else:
            src_points = points['fix_points_f5']
            dst_points = points['mov_points_f5_refined']
            im_shape = torch.tensor([x_fix_5.shape[4], x_fix_5.shape[3], x_fix_5.shape[2]],
                                    device=x_fix_5.device, dtype=x_fix_5.dtype)
        xfm_f2m = self._affine_from_points(src_points, dst_points)

        theta = self._affine_to_theta(xfm_f2m, im_shape)
        affine_para = self._theta_to_affine_para(theta)

        return affine_para

    @staticmethod
    def _flatten_tokens(x):
        b, c, d, h, w = x.shape
        tokens = einops.rearrange(x, 'b c d h w -> b (d h w) c')
        return tokens, (d, h, w)

    def _self_attention(self, tokens):
        x = self.sa_norm(tokens)
        x = self.attn_in(x)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.attn_out(x)
        return x

    def _cross_attention(self, q_tokens, kv_tokens):
        q = self.attn_in(self.ca_q_norm(q_tokens))
        kv = self.attn_in(self.ca_kv_norm(kv_tokens))
        x, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = self.attn_out(x)
        return x

    def _scores_from_tokens(self, tokens, shape):
        b = tokens.shape[0]
        d, h, w = shape
        score = self.score_mlp(tokens).squeeze(-1)
        return score.view(b, 1, d, h, w)

    @staticmethod
    def _tokens_to_map(tokens, shape):
        d, h, w = shape
        return einops.rearrange(tokens, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

    def extract_correspondence_points(self, x_fix_5, x_mov_5, x_fix_4=None, x_mov_4=None,
                                      x_fix_3=None, x_mov_3=None):
        # Step 1: only Fixed_5 receives self-attn and cross-attn residual.
        fix_tokens, shape_5 = self._flatten_tokens(x_fix_5)
        mov_tokens, _ = self._flatten_tokens(x_mov_5)

        f_tilde = fix_tokens + self._self_attention(fix_tokens)
        c_fix = self._cross_attention(f_tilde, mov_tokens)
        p_fix = f_tilde + self.gamma * c_fix

        score_fix = self._scores_from_tokens(p_fix, shape_5)
        cand_fix_f5 = self._grid_soft_candidates(score_fix)

        fix_feat_map = self._tokens_to_map(p_fix, shape_5)
        cand_mov_f5 = self._grid_correspondence_candidates(cand_fix_f5, fix_feat_map, x_mov_5)

        points = {
            'fix_points_f5': cand_fix_f5,
            'mov_points_f5_coarse': cand_mov_f5,
            'fix_points_f4': None,
            'mov_points_f4_coarse': None,
            'mov_points_f4_refined': None,
            'fix_points_f3': None,
            'mov_points_f3_coarse': None,
            'mov_points_f3_refined': None,
            'mu_c': None,
            'mu_f': None,
            'mu_total': None,
            'mu_c_f3': None,
            'mu_f_f3': None,
            'mu_total_f3': None,
            'mov_points_f5_refined': cand_mov_f5,
        }

        if x_fix_4 is None or x_mov_4 is None:
            return points

        # Step 2: map to F4 and refine moving points with a single 7-point search (radius=1).
        fix_points_f4 = self._map_points(cand_fix_f5, x_fix_5.shape[2:], x_fix_4.shape[2:])
        mov_points_f4_coarse = self._map_points(cand_mov_f5, x_mov_5.shape[2:], x_mov_4.shape[2:])
        mov_points_f4_refined, mu_c, mu_f = self._refine_points_f4(
            x_fix_4, x_mov_4, fix_points_f4, mov_points_f4_coarse
        )

        # Step 3: optionally continue the same template refinement on F3.
        mov_points_f5_refined = points['mov_points_f5_refined']
        if x_fix_3 is not None and x_mov_3 is not None:
            fix_points_f3 = self._map_points(fix_points_f4, x_fix_4.shape[2:], x_fix_3.shape[2:])
            mov_points_f3_coarse = self._map_points(mov_points_f4_refined, x_mov_4.shape[2:], x_mov_3.shape[2:])
            mov_points_f3_refined, mu_c_f3, mu_f_f3 = self._refine_points_f3(
                x_fix_3, x_mov_3, fix_points_f3, mov_points_f3_coarse
            )
        else:
            fix_points_f3 = None
            mov_points_f3_coarse = None
            mov_points_f3_refined = None
            mu_c_f3 = None
            mu_f_f3 = None
            # Map refined F4 points back to F5 coordinate space to keep downstream flow resolution unchanged.
            mov_points_f5_refined = self._map_points(mov_points_f4_refined, x_mov_4.shape[2:], x_mov_5.shape[2:])

        points['fix_points_f4'] = fix_points_f4
        points['mov_points_f4_coarse'] = mov_points_f4_coarse
        points['mov_points_f4_refined'] = mov_points_f4_refined
        points['fix_points_f3'] = fix_points_f3
        points['mov_points_f3_coarse'] = mov_points_f3_coarse
        points['mov_points_f3_refined'] = mov_points_f3_refined
        points['mu_c'] = mu_c
        points['mu_f'] = mu_f
        points['mu_total'] = mu_c + mu_f
        points['mu_c_f3'] = mu_c_f3
        points['mu_f_f3'] = mu_f_f3
        points['mu_total_f3'] = None if mu_c_f3 is None else mu_c_f3 + mu_f_f3
        points['mov_points_f5_refined'] = mov_points_f5_refined
        return points

    def _grid_correspondence_candidates(self, fixed_points, fix_feat_map, mov_feat_map):
        b, n, _ = fixed_points.shape
        _, c, d, h, w = mov_feat_map.shape

        fixed_desc = self._sample_points(fix_feat_map, fixed_points)
        fixed_desc = nnf.normalize(fixed_desc, dim=-1, eps=self.eps)

        mov_centers = self._map_points(fixed_points, fix_feat_map.shape[2:], mov_feat_map.shape[2:])
        p = self.patch_offsets_3.shape[0]
        offsets = self.patch_offsets_3.view(1, 1, p, 3).to(
            device=mov_feat_map.device,
            dtype=mov_feat_map.dtype,
        )
        mov_points = self._clamp_points(mov_centers.unsqueeze(2) + offsets, (d, h, w))

        mov_desc = self._sample_points(mov_feat_map, mov_points.reshape(b, n * p, 3))
        mov_desc = mov_desc.reshape(b, n, p, c)
        mov_desc = nnf.normalize(mov_desc, dim=-1, eps=self.eps)

        sim = (mov_desc * fixed_desc.unsqueeze(2)).sum(dim=-1)
        sim = sim / max(self.sinkhorn_tau, self.eps)
        probs = torch.softmax(sim, dim=-1)

        candidates = (probs.unsqueeze(-1) * mov_points).sum(dim=2)
        return self._clamp_points(candidates, (d, h, w))

    def _extract_local_patch(self, feat, centers):
        b, n, _ = centers.shape
        p = self.patch_offsets_3.shape[0]
        points = centers.unsqueeze(2) + self.patch_offsets_3.view(1, 1, p, 3).to(centers.dtype)
        points = self._clamp_points(points, feat.shape[2:])
        sampled = self._sample_points(feat, points.reshape(b, n * p, 3))
        return sampled.view(b, n, p, -1)

    def _extract_candidate_patches(self, feat, centers):
        b, n, k, _ = centers.shape
        p = self.patch_offsets_3.shape[0]
        points = centers.unsqueeze(3) + self.patch_offsets_3.view(1, 1, 1, p, 3).to(centers.dtype)
        points = self._clamp_points(points, feat.shape[2:])
        sampled = self._sample_points(feat, points.reshape(b, n * k * p, 3))
        return sampled.view(b, n, k, p, -1)

    def _search_offsets_with_templates(self, fix_patch, mov_feat, mov_centers, search_offsets):
        offsets = search_offsets.to(device=mov_centers.device, dtype=mov_centers.dtype)
        cand_centers = mov_centers.unsqueeze(2) + offsets.view(1, 1, -1, 3)
        cand_centers = self._clamp_points(cand_centers, mov_feat.shape[2:])

        mov_patch = self._extract_candidate_patches(mov_feat, cand_centers)
        fix_patch_n = nnf.normalize(fix_patch, dim=-1, eps=self.eps)
        mov_patch_n = nnf.normalize(mov_patch, dim=-1, eps=self.eps)

        sim = (fix_patch_n.unsqueeze(2) * mov_patch_n).sum(dim=-1).mean(dim=-1)
        sim = sim / max(self.sinkhorn_tau, self.eps)
        probs = torch.softmax(sim, dim=-1)

        offset = torch.matmul(probs, offsets)
        return offset

    def _refine_points_f4(self, x_fix_4, x_mov_4, fix_points_f4, mov_points_f4):
        fix_desc = self.desc_proj_4(x_fix_4)
        mov_desc = self.desc_proj_4(x_mov_4)

        fix_patch = self._extract_local_patch(fix_desc, fix_points_f4)
        mu_c = torch.zeros_like(mov_points_f4)
        mu_f = self._search_offsets_with_templates(fix_patch, mov_desc, mov_points_f4, self.offsets_r1)

        mov_refined = self._clamp_points(mov_points_f4 + mu_f, x_mov_4.shape[2:])
        return mov_refined, mu_c, mu_f

    def _refine_points_f3(self, x_fix_3, x_mov_3, fix_points_f3, mov_points_f3):
        fix_patch = self._extract_local_patch(x_fix_3, fix_points_f3)
        mu_c = torch.zeros_like(mov_points_f3)
        mu_f = self._search_offsets_with_templates(fix_patch, x_mov_3, mov_points_f3, self.offsets_r1)

        mov_refined = self._clamp_points(mov_points_f3 + mu_f, x_mov_3.shape[2:])
        return mov_refined, mu_c, mu_f

    def _grid_soft_candidates(self, score_map):
        b, _, d, h, w = score_map.shape
        g_d, g_h, g_w = self.grid_size

        pad_d = (g_d - d % g_d) % g_d
        pad_h = (g_h - h % g_h) % g_h
        pad_w = (g_w - w % g_w) % g_w

        padded = nnf.pad(score_map, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=-1e4)
        valid = torch.ones_like(score_map)
        valid = nnf.pad(valid, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0.0)

        d_p, h_p, w_p = padded.shape[2:]
        c_d = d_p // g_d
        c_h = h_p // g_h
        c_w = w_p // g_w

        candidates = []
        for i in range(g_d):
            for j in range(g_h):
                for k in range(g_w):
                    sd, ed = i * c_d, (i + 1) * c_d
                    sh, eh = j * c_h, (j + 1) * c_h
                    sw, ew = k * c_w, (k + 1) * c_w

                    cell_score = padded[:, 0, sd:ed, sh:eh, sw:ew].reshape(b, -1)
                    cell_valid = valid[:, 0, sd:ed, sh:eh, sw:ew].reshape(b, -1)
                    cell_score = cell_score.masked_fill(cell_valid < 0.5, -1e4)

                    weights = torch.softmax(cell_score, dim=-1) * cell_valid
                    weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)

                    ds = torch.arange(sd, ed, device=score_map.device, dtype=score_map.dtype)
                    hs = torch.arange(sh, eh, device=score_map.device, dtype=score_map.dtype)
                    ws = torch.arange(sw, ew, device=score_map.device, dtype=score_map.dtype)
                    gds, ghs, gws = torch.meshgrid(ds, hs, ws, indexing='ij')
                    coords = torch.stack([gds, ghs, gws], dim=-1).reshape(-1, 3)

                    point = torch.matmul(weights, coords)

                    valid_sum = cell_valid.sum(dim=-1, keepdim=True)
                    empty = (valid_sum < 0.5).expand(-1, 3)
                    center = torch.tensor([(sd + ed - 1) / 2.0,
                                           (sh + eh - 1) / 2.0,
                                           (sw + ew - 1) / 2.0],
                                          device=score_map.device,
                                          dtype=score_map.dtype).view(1, 3).expand(b, 3)
                    point = torch.where(empty, center, point)

                    point_d = point[:, 0:1].clamp(0, max(d - 1, 0))
                    point_h = point[:, 1:2].clamp(0, max(h - 1, 0))
                    point_w = point[:, 2:3].clamp(0, max(w - 1, 0))
                    point = torch.cat([point_d, point_h, point_w], dim=-1)
                    candidates.append(point)

        return torch.stack(candidates, dim=1)

    def _map_points(self, points, src_shape, dst_shape):
        src = torch.tensor(src_shape, device=points.device, dtype=points.dtype)
        dst = torch.tensor(dst_shape, device=points.device, dtype=points.dtype)
        scale = (dst - 1.0) / torch.clamp(src - 1.0, min=1.0)
        mapped = points * scale.view(1, 1, 3)
        mapped = self._clamp_points(mapped, dst_shape)
        return mapped

    def _clamp_points(self, points, shape):
        clamped = []
        for i, size in enumerate(shape):
            upper = float(max(size - 1, 0))
            clamped.append(points[..., i:i + 1].clamp(0.0, upper))
        return torch.cat(clamped, dim=-1)

    def _expand_neighbors(self, centers, offsets, shape):
        pts = centers.unsqueeze(2) + offsets.view(1, 1, -1, 3).to(centers.dtype)
        return self._clamp_points(pts, shape)

    def _sample_points(self, feat, points):
        b, c, d, h, w = feat.shape
        pts = self._clamp_points(points, (d, h, w))
        d_den = float(max(d - 1, 1))
        h_den = float(max(h - 1, 1))
        w_den = float(max(w - 1, 1))

        n_d = 2.0 * (pts[..., 0] / d_den) - 1.0
        n_h = 2.0 * (pts[..., 1] / h_den) - 1.0
        n_w = 2.0 * (pts[..., 2] / w_den) - 1.0
        grid = torch.stack([n_w, n_h, n_d], dim=-1).view(b, -1, 1, 1, 3)

        sampled = nnf.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled = sampled.squeeze(-1).squeeze(-1).transpose(1, 2)
        return sampled

    def _sample_nodes(self, feat, points):
        b, n, k, _ = points.shape
        flat_points = points.reshape(b, n * k, 3)
        sampled = self._sample_points(feat, flat_points)
        sampled = sampled.view(b, n, k, -1)
        sampled = nnf.normalize(sampled, dim=-1, eps=self.eps)
        return sampled

    @staticmethod
    def _pairwise_l2(node_feat):
        diff = node_feat.unsqueeze(-2) - node_feat.unsqueeze(-3)
        return torch.linalg.norm(diff, dim=-1)

    def _build_local_affinity_descriptors(self, feat, centers):
        centers = self._clamp_points(centers, feat.shape[2:])
        neigh_1 = self._expand_neighbors(centers, self.offsets_r1, feat.shape[2:])
        neigh_2 = self._expand_neighbors(centers, self.offsets_r2, feat.shape[2:])

        node_1 = self._sample_nodes(feat, neigh_1)
        node_2 = self._sample_nodes(feat, neigh_2)
        edge_1 = self._pairwise_l2(node_1)
        edge_2 = self._pairwise_l2(node_2)
        delta = nnf.normalize(node_2 - node_1, dim=-1, eps=self.eps)

        return {'node_1': node_1, 'node_2': node_2, 'edge_1': edge_1, 'edge_2': edge_2, 'delta': delta}

    def _matching_cost(self, desc_a, desc_b):
        a1 = desc_a['node_1']
        a2 = desc_a['node_2']
        b1 = desc_b['node_1']
        b2 = desc_b['node_2']

        cos_1 = torch.einsum('bikc,bjkc->bijk', a1, b1)
        cos_2 = torch.einsum('bikc,bjkc->bijk', a2, b2)
        l_node = ((1.0 - cos_1).sum(dim=-1) + (1.0 - cos_2).sum(dim=-1)) / 14.0

        e1_a = desc_a['edge_1']
        e2_a = desc_a['edge_2']
        e1_b = desc_b['edge_1']
        e2_b = desc_b['edge_2']
        edge_num = float(e1_a.shape[-1] * e1_a.shape[-2])
        l_edge = ((e1_a.unsqueeze(2) - e1_b.unsqueeze(1)).pow(2).sum(dim=(-1, -2)) +
                  (e2_a.unsqueeze(2) - e2_b.unsqueeze(1)).pow(2).sum(dim=(-1, -2))) / (2.0 * edge_num)

        d_a = desc_a['delta']
        d_b = desc_b['delta']
        l_cross = (d_a.unsqueeze(2) - d_b.unsqueeze(1)).pow(2).sum(dim=-1).mean(dim=-1)

        return l_node + l_edge + l_cross

    @staticmethod
    def _sinkhorn(score, iters):
        log_p = score
        for _ in range(iters):
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
        return torch.exp(log_p)

    @staticmethod
    def _greedy_assignment(cost_np):
        n_rows, n_cols = cost_np.shape
        used_cols = set()
        row_ind, col_ind = [], []
        for r in range(n_rows):
            sorted_cols = np.argsort(cost_np[r])
            picked = None
            for c in sorted_cols:
                c_int = int(c)
                if c_int not in used_cols:
                    picked = c_int
                    break
            if picked is None:
                break
            used_cols.add(picked)
            row_ind.append(r)
            col_ind.append(picked)
            if len(used_cols) >= n_cols:
                break
        return np.asarray(row_ind, dtype=np.int64), np.asarray(col_ind, dtype=np.int64)

    def _hungarian(self, cost):
        b, n, m = cost.shape
        hard = torch.zeros_like(cost)
        for idx in range(b):
            cost_np = cost[idx].detach().cpu().numpy()
            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost_np)
            else:
                row_ind, col_ind = self._greedy_assignment(cost_np)
            if len(row_ind) == 0:
                min_num = min(n, m)
                row_ind = np.arange(min_num, dtype=np.int64)
                col_ind = np.arange(min_num, dtype=np.int64)
            row_ind = torch.as_tensor(row_ind, device=cost.device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=cost.device, dtype=torch.long)
            hard[idx, row_ind, col_ind] = 1.0
        return hard

    def _weighted_affine_from_points(self, src_points_dhw, dst_points_dhw, weights):
        src_points = src_points_dhw[..., [2, 1, 0]]
        dst_points = dst_points_dhw[..., [2, 1, 0]]
        return self._solve_affine(src_points, dst_points, weights)

    def _affine_from_points(self, src_points_dhw, dst_points_dhw):
        src_points = src_points_dhw[..., [2, 1, 0]]
        dst_points = dst_points_dhw[..., [2, 1, 0]]
        return self._solve_affine_unweighted(src_points, dst_points)

    def _solve_affine(self, src_points, dst_points, weights):
        b, n, _ = src_points.shape
        w = torch.clamp(weights, min=self.eps)
        w = w / (w.sum(dim=-1, keepdim=True) + self.eps)

        ones = torch.ones((b, n, 1), device=src_points.device, dtype=src_points.dtype)
        src_h = torch.cat([src_points, ones], dim=-1)

        atw = src_h.transpose(1, 2) * w.unsqueeze(1)
        lhs = torch.matmul(atw, src_h)
        rhs = torch.matmul(atw, dst_points)

        eye = torch.eye(4, device=src_points.device, dtype=src_points.dtype).unsqueeze(0)
        lhs = lhs + 1e-4 * eye
        sol = torch.matmul(torch.linalg.pinv(lhs), rhs)

        affine = sol.transpose(1, 2)
        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=src_points.device, dtype=src_points.dtype)
        last_row = last_row.view(1, 1, 4).repeat(b, 1, 1)
        return torch.cat([affine, last_row], dim=1)

    def _solve_affine_unweighted(self, src_points, dst_points):
        b, n, _ = src_points.shape
        ones = torch.ones((b, n, 1), device=src_points.device, dtype=src_points.dtype)
        src_h = torch.cat([src_points, ones], dim=-1)
        pinv = torch.linalg.pinv(src_h)
        sol = torch.matmul(pinv, dst_points)
        affine = sol.transpose(1, 2)

        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=src_points.device, dtype=src_points.dtype)
        last_row = last_row.view(1, 1, 4).repeat(b, 1, 1)
        return torch.cat([affine, last_row], dim=1)

    def _affine_to_theta(self, xfm, im_shape):
        offset = (im_shape - 1) / 2.0
        offset = torch.clamp(offset, min=self.eps)
        A = xfm[:, :3, :3]
        b = xfm[:, :3, 3]

        A_scaled = A * offset
        A_norm = A_scaled / offset.view(1, 3, 1)
        b_norm = b / offset

        theta = torch.zeros((xfm.shape[0], 3, 4), device=xfm.device, dtype=xfm.dtype)
        theta[:, :3, :3] = A_norm
        theta[:, :3, 3] = b_norm
        return theta

    @staticmethod
    def _theta_to_affine_para(theta):
        identity = torch.zeros_like(theta)
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        identity[:, 2, 2] = 1
        return (theta - identity).reshape(theta.shape[0], 12)

    def _theta_to_flow(self, theta, src_shape):
        affine_grid = nnf.affine_grid(theta, src_shape, align_corners=True)
        shape = src_shape[2:]

        affine_grid = affine_grid[..., [2, 1, 0]]
        affine_grid = affine_grid.permute(0, 4, 1, 2, 3)
        scaled_channels = []
        for i in range(len(shape)):
            scale = float(max(shape[i] - 1, 1))
            scaled = (affine_grid[:, i:i + 1, ...] / 2.0 + 0.5) * scale
            scaled_channels.append(scaled)
        affine_grid = torch.cat(scaled_channels, dim=1)

        vectors = [torch.arange(0, s, device=theta.device, dtype=theta.dtype) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).unsqueeze(0)

        return affine_grid - grid


class AffineHead_block(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()

        self.affine_head_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.affine_head_2 = nn.Linear(in_channels // 2, 12, bias=False)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, x_in):
        x = torch.mean(x_in, dim=(2, 3, 4))
        x = self.affine_head_1(x)
        x = self.ReLU(x)
        x = self.affine_head_2(x)
        affine_para = self.Tanh(x)

        self.id = torch.zeros((1, 3, 4)).to(x_in.device)
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1
        affine_matrix = affine_para.reshape(1, 3, 4) + self.id
        affine_grid = nnf.affine_grid(affine_matrix, x_in.shape, align_corners=True)

        shape = x_in.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(x_in.device)

        affine_grid = affine_grid[..., [2, 1, 0]]
        affine_grid = affine_grid.permute(0, 4, 1, 2, 3)
        for i in range(len(shape)):
            affine_grid[:, i, ...] = (affine_grid[:, i, ...] / 2.0 + 0.5) * (shape[i] - 1)
        flow = affine_grid - grid

        return flow, affine_para


class PatchExpanding_block(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()

        self.up_conv = nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in):
        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention3D(nn.Module):

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        squeeze_channels = max(num_feat // squeeze_factor, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, squeeze_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(squeeze_channels, num_feat, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB3D(nn.Module):

    def __init__(self,
                 num_feat: int,
                 is_light_sr: bool = False,
                 compress_ratio: int = 3,
                 squeeze_factor: int = 30):
        super().__init__()
        hidden_feat = max(num_feat // compress_ratio, 1)
        if is_light_sr:
            self.cab = nn.Sequential(
                nn.Conv3d(num_feat, hidden_feat, kernel_size=1, stride=1, padding=0),
                nn.Conv3d(hidden_feat, hidden_feat, kernel_size=3, stride=1, padding=1, groups=hidden_feat),
                nn.GELU(),
                nn.Conv3d(hidden_feat, num_feat, kernel_size=1, stride=1, padding=0),
                nn.Conv3d(num_feat, num_feat, kernel_size=3, stride=1, padding=2, groups=num_feat, dilation=2),
                ChannelAttention3D(num_feat, squeeze_factor),
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv3d(num_feat, hidden_feat, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(hidden_feat, num_feat, kernel_size=3, stride=1, padding=1),
                ChannelAttention3D(num_feat, squeeze_factor),
            )

    def forward(self, x):
        return self.cab(x)


class DSConv3D(nn.Module):

    def __init__(self, channels: int, stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1,
                      groups=channels, bias=bias),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SS2D(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        del dt_rank, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, kwargs
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if Mamba is None:
            raise ImportError("MambaTrans_stage_block requires mamba-ssm. Install mamba-ssm before training this model.")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.scale_channels = max(self.d_inner // 2, 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.scale1_conv1 = DSConv3D(self.d_inner, stride=1, bias=conv_bias)
        self.scale1_conv2 = DSConv3D(self.d_inner, stride=1, bias=conv_bias)
        self.scale1_reduce = nn.Conv3d(self.d_inner, self.scale_channels, kernel_size=1, bias=bias,
                                       **factory_kwargs)
        self.scale1_norm = nn.LayerNorm(self.scale_channels)
        self.scale1_mamba = Mamba(d_model=self.scale_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scale1_restore = nn.Conv3d(self.scale_channels, self.d_inner, kernel_size=1, bias=bias,
                                        **factory_kwargs)

        self.scale2_conv1 = DSConv3D(self.d_inner, stride=2, bias=conv_bias)
        self.scale2_conv2 = DSConv3D(self.d_inner, stride=1, bias=conv_bias)
        self.scale2_reduce = nn.Conv3d(self.d_inner, self.scale_channels, kernel_size=1, bias=bias,
                                       **factory_kwargs)
        self.scale2_norm = nn.LayerNorm(self.scale_channels)
        self.scale2_mamba = Mamba(d_model=self.scale_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scale2_restore = nn.Conv3d(self.scale_channels, self.d_inner, kernel_size=1, bias=bias,
                                        **factory_kwargs)

        self.scale4_conv1 = DSConv3D(self.d_inner, stride=2, bias=conv_bias)
        self.scale4_conv2 = DSConv3D(self.d_inner, stride=2, bias=conv_bias)
        self.scale4_reduce = nn.Conv3d(self.d_inner, self.scale_channels, kernel_size=1, bias=bias,
                                       **factory_kwargs)
        self.scale4_norm = nn.LayerNorm(self.scale_channels)
        self.scale4_mamba = Mamba(d_model=self.scale_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.scale4_restore = nn.Conv3d(self.scale_channels, self.d_inner, kernel_size=1, bias=bias,
                                        **factory_kwargs)

        self.scale_fusion = nn.Conv3d(self.d_inner * 3, 3, kernel_size=1, bias=True, **factory_kwargs)
        self.scale_out_proj = nn.Conv3d(self.d_inner, self.d_inner, kernel_size=1, bias=bias, **factory_kwargs)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _run_scale_mamba(x, reduce, norm, mamba, restore):
        b, _, d, h, w = x.shape
        x = reduce(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = mamba(norm(x))
        x = x.transpose(1, 2).contiguous().view(b, -1, d, h, w)
        return restore(x)

    def forward_core(self, x: torch.Tensor):
        _, _, d, h, w = x.shape
        residual = x

        y1 = self.scale1_conv2(self.scale1_conv1(x))
        y1 = self._run_scale_mamba(
            y1, self.scale1_reduce, self.scale1_norm, self.scale1_mamba, self.scale1_restore
        )

        y2 = self.scale2_conv2(self.scale2_conv1(x))
        y2 = self._run_scale_mamba(
            y2, self.scale2_reduce, self.scale2_norm, self.scale2_mamba, self.scale2_restore
        )
        y2 = nnf.interpolate(y2, size=(d, h, w), mode='trilinear', align_corners=True)

        if min(d, h, w) < 8:
            y4 = y2
        else:
            y4 = self.scale4_conv2(self.scale4_conv1(x))
            y4 = self._run_scale_mamba(
                y4, self.scale4_reduce, self.scale4_norm, self.scale4_mamba, self.scale4_restore
            )
            y4 = nnf.interpolate(y4, size=(d, h, w), mode='trilinear', align_corners=True)

        scale_weights = torch.softmax(self.scale_fusion(torch.cat([y1, y2, y4], dim=1)), dim=1)
        y = (scale_weights[:, 0:1] * y1 +
             scale_weights[:, 1:2] * y2 +
             scale_weights[:, 2:3] * y4)
        return self.scale_out_proj(y) + residual

    def forward(self, x: torch.Tensor, **kwargs):
        del kwargs
        b, d, h, w, c = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        y = self.forward_core(x)
        y = y.permute(0, 2, 3, 4, 1).contiguous().view(b, d, h, w, -1)
        y = self.out_norm(y)
        y = y * nnf.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB3D(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        b, l, c = input.shape
        d, h, w = x_size
        if l != d * h * w:
            raise ValueError(f"VSSBlock token length {l} does not match x_size {x_size}.")

        input = input.view(b, d, h, w, c).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(
            self.ln_2(x).permute(0, 4, 1, 2, 3).contiguous()
        ).permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(b, -1, c).contiguous()
        return x


class MambaTrans_stage_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 drop_path: float = 0.0,
                 attn_drop: float = 0.0,
                 d_state: int = 16,
                 expand: float = 2.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.vss = VSSBlock(hidden_dim=embed_dim,
                            drop_path=drop_path,
                            attn_drop_rate=attn_drop,
                            d_state=d_state,
                            expand=expand)
        self.out_conv = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, stride=1, padding='same')

    def forward_part(self, x_in):
        b, c, d, h, w = x_in.shape
        x = einops.rearrange(x_in, 'b c d h w -> b (d h w) c')
        x = self.vss(x, (d, h, w))
        x = einops.rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        x_out = self.out_conv(x)
        return x_out

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_part, x_in, use_reentrant=False)
        else:
            x_out = self.forward_part(x_in)
        return x_out


class SwinTrans_stage_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    window_size=self.window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    use_checkpoint=use_checkpoint)
            self.blocks.append(block)

    def forward(self, x_in):

        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)

        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out


class SwinTrans_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MSA_block(embed_dim,
                              window_size=window_size,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim,
                             mlp_dim=int(embed_dim * mlp_ratio),
                             dropout_rate=drop)

    def forward_part1(self, x_in, mask_matrix):

        x = self.norm1(x_in)

        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):

        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):

        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix, use_reentrant=False)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)

        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x_out = x + self.forward_part2(x)

        return x_out


class MSA_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) *
                                                                     (2 * self.window_size[1] - 1) *
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):

        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)

        return x_out


class MLP_block(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.GELU = nn.GELU()

    def forward(self, x_in):
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x_out = self.drop2(x)

        return x_out


########################################################
# Functions
########################################################

def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  c)

    return windows


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
