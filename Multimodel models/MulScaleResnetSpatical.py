import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRBlock(nn.Module):
    """基础卷积-批归一化-ReLU模块，支持空洞卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):  # 新增dilation参数
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,  # 添加dilation参数
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    """空间注意力模块：突出重要空间区域特征"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化 (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大池化 (B, 1, H, W)
        attn = torch.cat([avg_out, max_out], dim=1)  # 拼接 (B, 2, H, W)
        attn = self.conv(attn)  # 卷积融合 (B, 1, H, W)
        return x * self.sigmoid(attn)  # 空间区域加权


class MultiScaleSpatialBlock(nn.Module):
    """多尺度空间特征提取模块"""

    def __init__(self, in_channels, out_channels, scales=[1, 3, 5]):
        super(MultiScaleSpatialBlock, self).__init__()
        self.scales = scales
        self.branches = nn.ModuleList()

        # 不同尺度的空间特征提取分支
        for scale in scales:
            if scale == 1:
                # 1x1卷积分支，捕捉局部细节
                self.branches.append(
                    CBRBlock(in_channels, out_channels, kernel_size=1, padding=0)
                )
            else:
                # 带空洞卷积的分支，扩大感受野
                self.branches.append(
                    CBRBlock(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=scale,  # 空洞卷积的padding应等于dilation
                        dilation=scale  # 现在CBRBlock支持dilation参数了
                    )
                )

        # 特征融合
        self.fusion = CBRBlock(
            out_channels * len(scales),
            out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        # 多尺度特征提取
        branch_outs = [branch(x) for branch in self.branches]

        # 特征融合
        fused = torch.cat(branch_outs, dim=1)
        fused = self.fusion(fused)

        return fused


class SpatialFeatureExtractor(nn.Module):
    """完整空间特征提取网络"""

    def __init__(self, input_channels, feature_dim=64, num_blocks=3):
        super(SpatialFeatureExtractor, self).__init__()
        # 输入通道调整
        self.input_adjust = CBRBlock(input_channels, feature_dim, kernel_size=1, padding=0)

        # 空间特征提取块（堆叠多个）
        self.spatial_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.spatial_blocks.append(nn.Sequential(
                MultiScaleSpatialBlock(feature_dim, feature_dim),  # 多尺度空间特征提取
                CBRBlock(feature_dim, feature_dim),  # 特征转换
                SpatialAttention()  # 空间注意力加权
            ))

        # 残差连接映射
        self.residual_conv = nn.Conv2d(input_channels, feature_dim, kernel_size=1, padding=0)

    def forward(self, x):
        # x shape: (B, C, H, W) 高光谱图像输入
        residual = self.residual_conv(x)  # 残差连接准备
        x = self.input_adjust(x)  # 输入通道调整

        # 空间特征提取
        for block in self.spatial_blocks:
            x = block(x) + x  # 残差连接

        # 最终融合
        x = x + residual
        return x


# 测试网络
if __name__ == "__main__":
    # 高光谱图像示例：批次大小=2，光谱通道=176，空间尺寸=30×30
    input_data = torch.randn(2, 176, 30, 30)

    # 初始化网络：输入176通道，输出64维特征，3个提取块
    model = SpatialFeatureExtractor(input_channels=176, feature_dim=64, num_blocks=3)

    # 前向传播
    output = model(input_data)
    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")  # 应保持空间维度，通道数为64
