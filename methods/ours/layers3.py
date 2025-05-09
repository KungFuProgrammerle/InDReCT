from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ops import ConvBNReLU, resize_to


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



# Conv-BN-ReLU2 Layer
class ConvBNReLU2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super(ConvBNReLU2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.use_relu:
            x = self.relu(x)
        return x


# CBAM (Channel and Spatial Attention Block)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        y = y.view(b, c)
        y = self.fc2(F.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


class AdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(AdaConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 初始化普通卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)

        # 额外的卷积层，用于根据输入特征图调整卷积核
        self.weight_transform = nn.Conv2d(in_channels, out_channels * kernel_size * kernel_size, kernel_size=1)

    def forward(self, x):
        # 获取输入特征图的尺寸
        batch_size, _, height, width = x.size()

        # 获取卷积核的基础权重
        weight = self.conv.weight  # 这个权重是通过初始的卷积层得到的
        kernel_size = self.kernel_size
        out_channels, in_channels, _, _ = weight.shape

        # 通过 weight_transform 来动态生成卷积核
        transformed_weight = self.weight_transform(x)  # 输出大小是 [batch_size, out_channels * kernel_size^2, height, width]

        # 重新调整 transformed_weight 的形状，使其与卷积核的形状匹配
        transformed_weight = transformed_weight.view(batch_size, out_channels, kernel_size, kernel_size, height, width)

        # 将 transformed_weight 应用于卷积操作
        # 由于我们是按 group 卷积计算，所以不需要修改输入数据
        out = F.conv2d(x, weight, padding=kernel_size // 2, groups=self.in_channels)

        # 返回经过自适应卷积后的结果
        return out

# MHSIU Module with Advanced Concepts
class MHSIU(nn.Module):
    def __init__(self, in_dim, num_groups=4, num_zoom=3):
        super(MHSIU, self).__init__()
        self.num_zoom = num_zoom

        # Convolution layers for each zoom level
        self.conv_l_pre = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)

        self.conv_l = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_m = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_s = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)  # intra-branch

        self.conv_lms = ConvBNReLU2(self.num_zoom * in_dim, 3 * in_dim, 1)  # inter-branch
        self.initial_merge = ConvBNReLU2(self.num_zoom * in_dim, 3 * in_dim, 1)  # inter-branch

        self.num_groups = num_groups
        self.trans = nn.Sequential(
            ConvBNReLU2(3 * in_dim // num_groups, in_dim // num_groups, 1),
            ConvBNReLU2(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

        # Attention mechanism (CBAM)
        self.attention_module = CBAM(in_dim*3)

        # Adaptive convolution for dynamic kernel adjustment
        self.adaconv = AdaConv(in_dim*3, in_dim*3, kernel_size=1)

        self.convList1 = nn.ModuleDict()
        self.convList2 = nn.ModuleDict()

        # Convolution layers for each zoom level
        for i in range(self.num_zoom):
            self.convList1[str(i)] = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)
            self.convList2[str(i)] = ConvBNReLU2(in_dim, in_dim, 3, 1, 1)

        self.out = ConvBNReLU2(in_dim*3, in_dim, 1)

    def forward(self, list):
        tgt_size = list[0].shape[2:]
        featureList = []

        for i in range(len(list)):
            l = self.convList1[str(i)](list[i])
            if l.shape[2:][0] > tgt_size[0]:
                l = resize_to(l, tgt_hw=tgt_size)
            elif l.shape[2:][0] < tgt_size[0]:
                l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
            l = self.convList2[str(i)](l)
            featureList.append(l)

        lms = torch.cat(featureList, dim=1)  # BT, 3C, H, W

        # Apply adaptive convolution for dynamic kernel adjustment
        lms = self.adaconv(lms)

        # Apply attention mechanism (CBAM)
        attn = self.attention_module(lms)

        x = self.out(attn)  # Final output after attention

        return x
