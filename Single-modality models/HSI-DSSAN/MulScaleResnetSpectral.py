import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRBlock(nn.Module):
    """基础卷积-批归一化-ReLU模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpectralAttention(nn.Module):
    """光谱注意力模块：突出重要波段特征"""

    def __init__(self, in_channels, reduction=4):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 空间维度压缩
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        attn = self.avg_pool(x)  # (B, C, 1, 1)
        attn = self.fc(attn)  # (B, C, 1, 1)
        return x * attn  # 光谱通道加权


class SpectralConvBlock(nn.Module):
    """光谱卷积模块：使用1D卷积捕捉通道间相关性"""

    def __init__(self, in_channels, kernel_sizes=[3, 5, 7]):
        super(SpectralConvBlock, self).__init__()
        self.branches = nn.ModuleList()

        # 多尺度1D卷积分支（不同 kernel size 捕捉不同范围的光谱关联）
        for kernel_size in kernel_sizes:
            self.branches.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=in_channels  # 深度可分离卷积，降低计算量
                )
            )

        # 特征融合
        self.fusion = nn.Conv2d(in_channels * len(kernel_sizes), in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # 维度重塑以适配1D卷积: (B*H*W, C, 1)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, C, 1)

        # 多尺度光谱特征提取
        branch_outs = []
        for branch in self.branches:
            out = branch(x_reshaped)  # (B*H*W, C, 1)
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # 恢复为 (B, C, H, W)
            branch_outs.append(out)

        # 特征融合
        fused = torch.cat(branch_outs, dim=1)  # (B, C*K, H, W) K为分支数
        fused = self.fusion(fused)  # (B, C, H, W)

        return fused


class SpectralFeatureExtractor(nn.Module):
    """完整光谱特征提取网络"""

    def __init__(self, input_channels, feature_dim=64, num_blocks=3):
        super(SpectralFeatureExtractor, self).__init__()
        # 输入通道调整
        self.input_adjust = CBRBlock(input_channels, feature_dim, kernel_size=1, padding=0)

        # 光谱特征提取块（堆叠多个）
        self.spectral_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.spectral_blocks.append(nn.Sequential(
                SpectralConvBlock(feature_dim),  # 1D卷积提取光谱关联
                CBRBlock(feature_dim, feature_dim),  # 特征转换
                SpectralAttention(feature_dim)  # 光谱注意力加权
            ))

        # 残差连接映射（当输入输出通道不同时使用）
        self.residual_conv = nn.Conv2d(input_channels, feature_dim, kernel_size=1, padding=0)

    def forward(self, x):
        # x shape: (B, C, H, W) 高光谱图像输入
        residual = self.residual_conv(x)  # 残差连接准备
        x = self.input_adjust(x)  # 输入通道调整

        # 光谱特征提取
        for block in self.spectral_blocks:
            x = block(x) + x  # 残差连接

        # 最终融合
        x = x + residual
        return x


# 测试网络
if __name__ == "__main__":
    # 高光谱图像示例：批次大小=2，光谱通道=176，空间尺寸=30×30
    input_data = torch.randn(2, 176, 30, 30)

    # 初始化网络：输入176通道，输出64维特征，3个提取块
    model = SpectralFeatureExtractor(input_channels=176, feature_dim=64, num_blocks=3)

    # 前向传播
    output = model(input_data)
    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")  # 应保持空间维度，通道数为64
