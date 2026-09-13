import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入之前定义的特征提取器
from MulScaleResnetSpectral import SpectralFeatureExtractor
from MulScaleResnetSpatical import SpatialFeatureExtractor



# 复用之前定义的基础模块
class CBRBlock(nn.Module):
    """基础卷积-批归一化-ReLU模块，支持空洞卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CrossAttention(nn.Module):
    """交叉注意力模块：捕捉光谱和空间特征之间的关联"""

    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 注意力权重因子
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # x: 光谱特征 (B, C, H, W)
        # y: 空间特征 (B, C, H, W)
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        proj_key = self.key_conv(y).view(B, -1, H * W)  # (B, C//8, H*W)

        # 计算光谱-空间关联注意力图
        energy = torch.bmm(proj_query, proj_key)  # (B, H*W, H*W)
        attention = self.softmax(energy)  # (B, H*W, H*W)

        proj_value = self.value_conv(y).view(B, -1, H * W)  # (B, C, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)  # (B, C, H, W)

        return self.gamma * out + x  # 残差连接


class AdaptiveFusion(nn.Module):
    """自适应特征融合模块：动态学习光谱和空间特征的权重"""

    def __init__(self, feature_dim):
        super(AdaptiveFusion, self).__init__()
        # 特征重要性权重学习
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 2, kernel_size=1),  # 输出2个权重（光谱和空间）
            nn.Softmax(dim=1)  # 权重归一化
        )

        # 交叉注意力模块（双向）
        self.cross_attn_spectral = CrossAttention(feature_dim)  # 空间特征增强光谱特征
        self.cross_attn_spatial = CrossAttention(feature_dim)  # 光谱特征增强空间特征

        # 融合后特征精炼
        self.refine = CBRBlock(feature_dim, feature_dim, kernel_size=3)

    def forward(self, spectral_feat, spatial_feat):
        # 步骤1：交叉注意力增强
        spectral_enhanced = self.cross_attn_spectral(spectral_feat, spatial_feat)
        spatial_enhanced = self.cross_attn_spatial(spatial_feat, spectral_feat)

        # 步骤2：自适应权重学习
        concat_feat = torch.cat([spectral_enhanced, spatial_enhanced], dim=1)  # (B, 2C, H, W)
        weights = self.attention(concat_feat)  # (B, 2, H, W)：2个权重分别对应光谱和空间

        # 步骤3：加权融合
        fused = spectral_enhanced * weights[:, 0:1, :, :] + spatial_enhanced * weights[:, 1:2, :, :]

        # 步骤4：特征精炼
        fused = self.refine(fused)
        return fused


class RegressionHead(nn.Module):
    """回归任务头：将融合特征映射到回归目标"""

    def __init__(self, feature_dim, output_dim=1):
        super(RegressionHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, output_dim)
        )


    def forward(self, x):
        # x: 融合特征 (B, C, H, W)
        x = self.global_pool(x).view(x.size(0), -1)  # (B, C)
        x = self.fc(x)  # (B, output_dim)
        return x


class SpectralSpatialRegressionNetwork(nn.Module):
    """完整的光谱-空间特征融合回归网络"""

    def __init__(self, spectral_extractor, spatial_extractor, feature_dim=64, output_dim=1):
        super(SpectralSpatialRegressionNetwork, self).__init__()
        self.spectral_extractor = spectral_extractor  # 光谱特征提取器
        self.spatial_extractor = spatial_extractor  # 空间特征提取器
        self.fusion_module = AdaptiveFusion(feature_dim)  # 特征融合模块
        self.regression_head = RegressionHead(feature_dim, output_dim)  # 回归头

    '''
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 (B, C, 1, 1)

    def getFeatures(self, x):
        spectral_feat = self.spectral_extractor(x)  # 光谱特征 (B, C, H, W)
        spatial_feat = self.spatial_extractor(x)  # 空间特征 (B, C, H, W)

        # 特征融合
        x = self.fusion_module(spectral_feat, spatial_feat)  # (B, C, H, W)

        x = self.global_pool(x).view(x.size(0), -1)  # (B, C)

        return  x
    '''

    def forward(self, x):
        # x: 输入高光谱图像 (B, C, H, W)
        spectral_feat = self.spectral_extractor(x)  # 光谱特征 (B, C, H, W)
        spatial_feat = self.spatial_extractor(x)  # 空间特征 (B, C, H, W)

        # 特征融合
        fused_feat = self.fusion_module(spectral_feat, spatial_feat)  # (B, C, H, W)

        # 回归预测
        output = self.regression_head(fused_feat)  # (B, output_dim)
        return output


class SpectralSpatialRegressionNetworkGetFs(nn.Module):
    """完整的光谱-空间特征融合回归网络"""

    def __init__(self, spectral_extractor, spatial_extractor, feature_dim=64, output_dim=1):
        super(SpectralSpatialRegressionNetworkGetFs, self).__init__()
        self.spectral_extractor = spectral_extractor  # 光谱特征提取器
        self.spatial_extractor = spatial_extractor  # 空间特征提取器
        self.fusion_module = AdaptiveFusion(feature_dim)  # 特征融合模块
        self.regression_head = RegressionHead(feature_dim, output_dim)  # 回归头

    '''
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 (B, C, 1, 1)

    def getFeatures(self, x):
        spectral_feat = self.spectral_extractor(x)  # 光谱特征 (B, C, H, W)
        spatial_feat = self.spatial_extractor(x)  # 空间特征 (B, C, H, W)

        # 特征融合
        x = self.fusion_module(spectral_feat, spatial_feat)  # (B, C, H, W)

        x = self.global_pool(x).view(x.size(0), -1)  # (B, C)

        return  x
    '''

    def getFeatures(self, x):

        # x: 输入高光谱图像 (B, C, H, W)
        spectral_feat = self.spectral_extractor(x)  # 光谱特征 (B, C, H, W)
        spatial_feat = self.spatial_extractor(x)  # 空间特征 (B, C, H, W)

        # 特征融合
        fused_feat = self.fusion_module(spectral_feat, spatial_feat)  # (B, C, H, W)
        return fused_feat


    def forward(self, x):
        # x: 输入高光谱图像 (B, C, H, W)
        spectral_feat = self.spectral_extractor(x)  # 光谱特征 (B, C, H, W)
        spatial_feat = self.spatial_extractor(x)  # 空间特征 (B, C, H, W)

        # 特征融合
        fused_feat = self.fusion_module(spectral_feat, spatial_feat)  # (B, C, H, W)

        # 回归预测
        output = self.regression_head(fused_feat)  # (B, output_dim)
        return output


'''
# 测试代码
if __name__ == "__main__":

    # 初始化光谱和空间特征提取器
    spectral_extractor = SpectralFeatureExtractor(input_channels=176, feature_dim=64)
    spatial_extractor = SpatialFeatureExtractor(input_channels=176, feature_dim=64)

    # 初始化融合回归网络
    model = SpectralSpatialRegressionNetwork(
        spectral_extractor=spectral_extractor,
        spatial_extractor=spatial_extractor,
        feature_dim=64,
        output_dim=1  # 假设回归目标是单值（如生物量、温度等）
    )

    # 测试输入
    input_data = torch.randn(2, 176, 30, 30)  # (B=2, C=176, H=30, W=30)
    output = model(input_data)

    print(f"输入形状: {input_data.shape}")
    print(f"回归输出形状: {output.shape}")  # 应输出 (2, 1)

'''