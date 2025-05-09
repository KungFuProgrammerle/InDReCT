import torch
from torch import nn

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


class MultiScaleAdaptiveDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3, 5], kernel_sizes=[3, 3, 3, 3],
                ):
        super(MultiScaleAdaptiveDilatedConv, self).__init__()
        self.dilation_rates = dilation_rates
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels

        # 多扩张率卷积分支
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=d, dilation=d, bias=True)
            for d, k in zip(dilation_rates, kernel_sizes)
        ])

        # 可学习的梯度特征提取器
        self.learnable_gradient_extractor = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, bias=False)

        # Attention模块
        self.attention = nn.Sequential(
            nn.Conv2d(4, len(dilation_rates), kernel_size=1),  # Attention on gradient map
            nn.ReLU(),
            nn.Softmax(dim=1)  # Attention across the branches (scales)
        )
        self.weight_mask=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=1, bias=False),
            nn.Conv2d(61, 1, kernel_size=1, stride=1, padding=1, bias=False),

        )

    def forward(self, x, data):
        # 获取输入图像的长宽信息
        _, _, h, w = x.shape  # (batch_size, channels, height, width)

        # 将长宽信息作为输入，构造一个与输入图像大小相同的张量
        spatial_info = torch.tensor([h, w], dtype=torch.float32).to(x.device)  # (2,) 长宽信息

        # 扩展为批次大小和空间大小匹配的形状
        spatial_info = spatial_info.view(1, 2, 1, 1).expand(x.size(0), 2, h, w)

        # # 生成缩放系数的引导参数
        # scale_factor = torch.tensor(factor, dtype=torch.float32).to(x.device)  # 缩放系数
        # scale_info = scale_factor.view(1, 1, 1, 1).expand(x.size(0), 1, h, w)

        # 多分支卷积输出
        branch_outputs = [branch(x) for branch in self.branches]

        # 使用Learnable梯度网络提取梯度信息
        learnable_grad = torch.abs(self.learnable_gradient_extractor(x))

        # 注意力机制生成权重
        weights = self.attention(learnable_grad)

        # 将空间信息和缩放系数作为附加输入，影响注意力机制的计算
        # 将空间信息和缩放系数与注意力权重相乘，或者进行其他操作
        #* spatial_info[:, 0:1, :, :] * scale_info
        weights = weights   # 空间信息和缩放系数的影响

        # 对每个分支输出进行加权
        train = data["train"][0]
        if train:
            curr = data["curr"]
            if (curr>0.5):
                curr = 0.5
        else:
            curr=0.5
        # mask = data["mask"]
        # mask = resize_to(mask,(h,w))
        # else:
        #     f=h
        #     mask= torch.ones(h,w).cuda()
        weighted_outputs = [
            branch_output * weights[:, i:i + 1, :, :] *h*(2-curr)# # 对每个分支输出加权,传入特征图 或mask的大小（大小/384=系数）
            for i, branch_output in enumerate(branch_outputs)
        ]

        # 对加权后的输出进行求和
        output = torch.sum(torch.stack(weighted_outputs, dim=1), dim=1)  # 对不同分支的输出进行加权求和

        return output


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, dilation_rates=[1,2,3,5]):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(

            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.ad = MultiScaleAdaptiveDilatedConv(inplanes,planes,dilation_rates=dilation_rates)

    def forward(self, x,factor):

        x = self.ad(x,factor)
        x = self.block(x)
        return x



class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class att(nn.Module):
    def __init__(self, channels,  factor=16):#factorc 32

        super(att, self).__init__()
        self.conv = ConvBNR(48, 16, 3)
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)


    def forward(self, x):
        x=self.conv(x)
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        x1 = self.gn(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights =  torch.matmul(x11, x22).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
class ConvBNR2(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0,dilation=1, bias=False):
        super(ConvBNR2, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBNR2(inplanes, 32, 1),
            ConvBNR2(32, 32, kernel_size,padding=1,stride=stride,dilation=dilation,bias=bias),
            ConvBNR2(32, planes, 1)
        )


    def forward(self, x):

        x2=self.block(x)

        if x2.size()[2:] != x.size()[2:]:
            x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=False)

        x=x+x2
        return x



class BGM(nn.Module):
    def __init__(self):
        super(BGM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 64)
        self.reduce3 = Conv1x1(1024, 64)
        self.reduce4 = Conv1x1(2048, 64)
        self.block00=nn.Sequential(
            ConvBNR(256, 64, 3),

        )

        self.block0=nn.Sequential(
            ConvBlock(64,64)

        )
        self.block1=nn.Sequential(

            ConvBlock(64, 64,stride=2),
        )
        self.block2=nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block3 = nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block4=nn.Sequential(

            # ConvBNR2(64, 64, 1),
            # ConvBNR2(64, 64, 3, padding=1, stride=1),
            # ConvBNR2(64, 1, 1)
            ConvBlock(64, 64, stride=1),
            nn.Conv2d(64, 1, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, x4,x3,x2, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)

        out = torch.concat((x4,x3,x2, x1), dim=1)
        out=self.block00(out)
        out0=self.block0(out)

        out1=self.block1(out)

        out2=self.block2(out)
        out3=self.block3(out)
        out1=F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        out2=F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
        out3=F.interpolate(out3, scale_factor=8, mode='bilinear', align_corners=False)

        # out=torch.concat((out0,out1, out2, out3), 1)

        out=torch.add(torch.add(torch.add(out0,out1),out2),out3)
        oout = self.block4(out)


        return oout


class GCAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(GCAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, dilation_rates=[2,3,5,7])
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, dilation_rates=[2,3,5,7])
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, dilation_rates=[2,3,5,7])
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, dilation_rates=[2,3,5,7])

        self.conv1_2 = Conv1x1(channel, channel) #ConvBlock(channel, channel)

        self.conv3_3 = ConvBNR(channel, channel , dilation_rates=[2,3,5,7])

        self.block1=ConvBlock(channel//4,channel//4)


    def forward(self, lf, hf,factor):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1],factor)

        x1 = self.dconv5_1(xc[1] + x0 + xc[2],factor)

        x2 = self.dconv7_1(xc[2] + x1 + xc[3],factor)

        x3 = self.dconv9_1(xc[3] + x2,factor)
        x33 = self.block1(x3)
        x22 = self.block1(x33+x2)
        x11 = self.block1(x1+x22)
        x00 = self.block1(x0+x11)
        xx = self.conv1_2(torch.cat((x00, x11, x22, x33), dim=1))
        x = self.conv3_3(x + xx,factor)

        return x

class BGM2(nn.Module):
    def __init__(self):
        super(BGM2, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 64)
        self.reduce3 = Conv1x1(320, 64)
        self.reduce4 = Conv1x1(512, 64)
        self.block00=nn.Sequential(
            ConvBNR(256, 64, 3),

        )

        self.block0=nn.Sequential(
            ConvBlock(64,64)

        )
        self.block1=nn.Sequential(

            ConvBlock(64, 64,stride=2),
        )
        self.block2=nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block3 = nn.Sequential(

            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
            ConvBlock(64, 64,stride=2),
        )
        self.block4=nn.Sequential(
            # ConvBNR(256, 1, 3)
            ConvBNR2(64, 64, 1),
            ConvBNR2(64, 64, 3, padding=1, stride=1),
            ConvBNR2(64, 1, 1)
        )

    def forward(self, x4, x3, x2, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)

        out = torch.concat((x4, x3, x2, x1), dim=1)
        out = self.block00(out)
        out0 = self.block0(out)

        out1 = self.block1(out)

        out2 = self.block2(out)
        out3 = self.block3(out)
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, scale_factor=8, mode='bilinear', align_corners=False)

        # out=torch.concat((out0,out1, out2, out3), 1)

        out = torch.add(torch.add(torch.add(out0, out1), out2), out3)
        oout = self.block4(out)

        return oout



if __name__=="__main__":
    moudel=att(32)
    out=moudel(torch.randn(16,32,224,224))
    print(out.shape)

