from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ops import ConvBNReLU, resize_to
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False), # 降维
#             nn.ReLU(),
#             nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False) # 升维
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))  # 平均池化路径
#         max_out = self.fc(self.max_pool(x))  # 最大池化路径
#         out = avg_out + max_out
#         return self.sigmoid(out) * x
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  # 7x7卷积
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 平均池化和最大池化在通道维度合并
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
#         max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
#         out = torch.cat([avg_out, max_out], dim=1)  # 在通道维度拼接
#         out = self.conv(out)
#         return self.sigmoid(out) * x
#
#
# class ReverseAttention(nn.Module):
#     def __init__(self):
#         super(ReverseAttention, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, attention):
#         reverse_attention = 1 - attention
#         return x * reverse_attention
#
#
# class EnhancedAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16, kernel_size=7):
#         super(EnhancedAttention, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)
#         self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
#         self.reverse_attention = ReverseAttention()
#
#     def forward(self, x):
#         ca_out = self.channel_attention(x)
#         sa_out = self.spatial_attention(x)
#
#         # 加权融合
#         fusion_weight = torch.sigmoid(self.fusion_conv(torch.cat([ca_out, sa_out], dim=1)))
#         out = fusion_weight * ca_out + (1 - fusion_weight) * sa_out
#
#         # 生成反向注意力并应用
#         spatial_attention = self.spatial_attention(x)
#         out = self.reverse_attention(out, spatial_attention)
#
#         return out

class GatedAttentionUnit(nn.Module):
    def __init__(self, embed_dim):
        super(GatedAttentionUnit, self).__init__()
        self.proj_y = nn.Linear(embed_dim * 2, embed_dim)  # 拼接后的维度是 2 * embed_dim
        self.proj_x = nn.Linear(embed_dim, embed_dim)  # 门控输入是 embed_dim
        self.activation = nn.Sigmoid()
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, content, gate_source):
        # 将 content 和 gate_source 拼接在最后一个维度
        content = self.proj_y(torch.cat((content, gate_source), dim=-1))

        # 生成门控信号
        gate_signal = self.proj_x(gate_source)
        gate = self.activation(gate_signal)  # Sigmoid生成门控信号

        # 内容和门控动态加权
        out = content * gate
        out = self.output_proj(out)  # 输出映射
        return out


class GatedCrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(GatedCrossModalAttention, self).__init__()
        self.gau = GatedAttentionUnit(embed_dim)

    def forward(self, q, kv):
        B, C, H, W = q.shape

        # 将输入展平为序列形式
        q = q.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        kv = kv.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 使用 GAU 进行跨模态建模 (q 作为内容，kv 作为门控)
        out = self.gau(q, kv)  # [B, H*W, C]

        # 恢复到图像维度
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out


class CascadeMultiModalAttention(nn.Module):
    def __init__(self, embed_dim, num_modalities):
        super(CascadeMultiModalAttention, self).__init__()
        self.num_modalities = num_modalities
        self.attention_modules = nn.ModuleList([
            GatedCrossModalAttention(embed_dim)
            for _ in range(num_modalities)
        ])

    def forward(self, rgb, modalities):
        aligned_feat = rgb
        for i in range(self.num_modalities):
            # rgb 作为内容，modalities[i] 作为门控
            aligned_feat = self.attention_modules[i](aligned_feat, modalities[i])
        return aligned_feat


class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        """A simple ASPP variant.

        Args:
            in_dim (int): Input channels.
            out_dim (int): Output channels.
            dilation (int, optional): Dilation of the convolution operation. Defaults to 3.
        """
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)

        # dilation branch
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        # global branch
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))


class DifferenceAwareOps(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.temperal_proj_norm = nn.LayerNorm(num_frames, elementwise_affine=False)
        self.temperal_proj_kv = nn.Linear(num_frames, 2 * num_frames, bias=False)
        self.temperal_proj = nn.Sequential(
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_frames, num_frames, 3, 1, 1, bias=False),
        )
        for t in self.parameters():
            nn.init.zeros_(t)

    def forward(self, x):
        if self.num_frames == 1:
            return x

        unshifted_x_tmp = rearrange(x, "(b t) c h w -> b c h w t", t=self.num_frames)
        B, C, H, W, T = unshifted_x_tmp.shape
        shifted_x_tmp = torch.roll(unshifted_x_tmp, shifts=1, dims=-1)
        diff_q = shifted_x_tmp - unshifted_x_tmp  # B,C,H,W,T
        diff_q = self.temperal_proj_norm(diff_q)  # normalization along the time

        # merge all channels
        diff_k, diff_v = self.temperal_proj_kv(diff_q).chunk(2, dim=-1)
        diff_qk = torch.einsum("bxhwt, byhwt -> bxyt", diff_q, diff_k) * (H * W) ** -0.5
        temperal_diff = torch.einsum("bxyt, byhwt -> bxhwt", diff_qk.softmax(dim=2), diff_v)

        temperal_diff = rearrange(temperal_diff, "b c h w t -> (b c) t h w")
        shifted_x_tmp = self.temperal_proj(temperal_diff)  # combine different time step
        shifted_x_tmp = rearrange(shifted_x_tmp, "(b c) t h w -> (b t) c h w", c=x.shape[1])
        return x + shifted_x_tmp


class RGPU(nn.Module):
    def __init__(self, in_c, num_groups=6, hidden_dim=None, num_frames=1):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(
            DifferenceAwareOps(num_frames=num_frames),
            ConvBNReLU(num_groups * hidden_dim, in_c, 3, 1, 1, act_name=None),
        )
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []
        gates = []

        group_id = 0
        curr_x = xs[group_id]
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        for group_id in range(1, self.num_groups - 1):
            curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
            branch_out = self.interact[str(group_id)](curr_x)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

        group_id = self.num_groups - 1
        curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_gate = branch_out.chunk(2, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        out = torch.cat(outs, dim=1)
        gate = self.gate_genator(torch.cat(gates, dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4,num_zoom=3):
        super().__init__()
        self.num_zoom=num_zoom




        # self.conv_lms = ConvBNReLU(self.num_zoom * in_dim, 3 * in_dim, 1)  # inter-branch



        self.num_groups = num_groups
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )
        self.convList1 = nn.ModuleDict()
        self.convList2 = nn.ModuleDict()
        for i in range (self.num_zoom):
            self.convList1[str(i)]=ConvBNReLU(in_dim, in_dim, 3, 1, 1)
            self.convList2[str(i)]=ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.out = ConvBNReLU(in_dim, in_dim, 1)
        # self.att = EnhancedAttention(3*in_dim)
        self.att = CascadeMultiModalAttention(embed_dim=in_dim,  num_modalities=self.num_zoom-1)

    def forward(self, list):
        tgt_size = list[0].shape[2:]
        featureList=[]
        for i in range (len(list)):
            l = self.convList1[str(i)](list[i])
            if (l.shape[2:][0]>tgt_size[0]):
                l=resize_to(l, tgt_hw=tgt_size)
            elif (l.shape[2:][0]<tgt_size[0]):
                l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
            l = self.convList2[str(i)](l)
            featureList.append(l)


        # lms = torch.cat(featureList, dim=1)  # BT,3C,H,W
        # lms = torch.sum(torch.stack(featureList), dim=0)

        attn = self.att(featureList[0],featureList[1:])
        # attn = self.conv_lms(attn)  # BT,3C,H,W
        # attn  =self.att(attn)
        x = self.out(attn )




        return x
