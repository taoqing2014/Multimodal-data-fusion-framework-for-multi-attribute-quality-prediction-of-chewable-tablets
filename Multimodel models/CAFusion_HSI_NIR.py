import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention1D(nn.Module):
    """一维数据交叉注意力模块：捕捉两组特征间的关联"""

    def __init__(self, feature_dim):
        super(CrossAttention1D, self).__init__()
        # 降维投影（减少计算量）
        self.query_proj = nn.Linear(feature_dim, feature_dim // 8)
        self.key_proj = nn.Linear(feature_dim, feature_dim // 8)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # 注意力权重因子
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # x: 第一组特征 (B, N)
        # y: 第二组特征 (B, N)

        # 投影为查询、键、值
        query = self.query_proj(x)  # (B, N//8)
        key = self.key_proj(y)  # (B, N//8)
        value = self.value_proj(y)  # (B, N)

        # 计算注意力分数（关联度）
        energy = torch.bmm(query.unsqueeze(1), key.unsqueeze(2)).squeeze()  # (B,) → 简化为标量关联度
        attention = self.softmax(energy.unsqueeze(1))  # (B, 1) → 归一化

        # 用注意力加权值特征，并与原始特征残差连接
        out = value * attention  # (B, N) → 按样本加权
        return self.gamma * out + x  # 残差连接增强原始特征


class AdaptiveFusion1D(nn.Module):
    """一维数据自适应融合模块：动态学习两组特征的权重"""

    def __init__(self, feature_dim):
        super(AdaptiveFusion1D, self).__init__()
        # 双向交叉注意力
        self.cross_attn_1 = CrossAttention1D(feature_dim)  # 用第二组特征增强第一组
        self.cross_attn_2 = CrossAttention1D(feature_dim)  # 用第一组特征增强第二组

        # 自适应权重学习网络
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),  # 输出两组特征的权重
            nn.Softmax(dim=1)  # 权重归一化
        )

        # 融合后特征精炼
        self.refine = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        # feat1: 第一组特征 (B, N)
        # feat2: 第二组特征 (B, N)

        # 步骤1：交叉注意力增强
        feat1_enhanced = self.cross_attn_1(feat1, feat2)  # 用feat2增强feat1
        feat2_enhanced = self.cross_attn_2(feat2, feat1)  # 用feat1增强feat2

        # 步骤2：学习自适应权重
        concat_feat = torch.cat([feat1_enhanced, feat2_enhanced], dim=1)  # (B, 2N)
        weights = self.weight_net(concat_feat)  # (B, 2) → 每组特征的权重

        # 步骤3：加权融合
        fused = feat1_enhanced * weights[:, 0:1] + feat2_enhanced * weights[:, 1:2]  # (B, N)

        # 步骤4：特征精炼
        fused = self.refine(fused)
        return fused


# 测试代码
if __name__ == "__main__":
    # 配置
    B = 32  # 样本数
    N = 128  # 特征维度

    # 生成测试数据（两组一维特征）
    feat1 = torch.randn(B, N)  # 第一组特征
    feat2 = torch.randn(B, N)  # 第二组特征

    # 初始化融合模块
    fusion_module = AdaptiveFusion1D(feature_dim=N)

    # 执行融合
    fused_feat = fusion_module(feat1, feat2)

    # 输出形状检查
    print(f"输入特征1形状: {feat1.shape}")
    print(f"输入特征2形状: {feat2.shape}")
    print(f"融合后特征形状: {fused_feat.shape}")  # 应输出 (32, 128)