import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionForBN(nn.Module):
    """
    适用于B×N形状特征的交叉注意力模块
    输入: 两个模态的特征，形状均为 (batch_size, feature_dim)
    输出: 融合后的特征，形状为 (batch_size, out_dim)
    """

    def __init__(self, feature_dim, num_heads=4, out_dim=None, dropout=0.1):
        """
        参数:
            feature_dim: 输入特征的维度 (N)
            num_heads: 注意力头的数量
            out_dim: 输出特征的维度，默认与feature_dim相同
            dropout: dropout概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim if out_dim is not None else feature_dim
        self.num_heads = num_heads
        self.head_dim = self.out_dim // num_heads

        # 确保输出维度可以被注意力头数量整除
        assert self.head_dim * num_heads == self.out_dim, "输出维度必须是注意力头数量的整数倍"

        # 线性变换层，将输入特征映射到Q, K, V
        # 对于模态A作为查询，模态B作为键和值
        self.q_proj_a = nn.Linear(feature_dim, self.out_dim)  # 模态A -> 查询Q
        self.k_proj_b = nn.Linear(feature_dim, self.out_dim)  # 模态B -> 键K
        self.v_proj_b = nn.Linear(feature_dim, self.out_dim)  # 模态B -> 值V

        # 对于模态B作为查询，模态A作为键和值
        self.q_proj_b = nn.Linear(feature_dim, self.out_dim)  # 模态B -> 查询Q
        self.k_proj_a = nn.Linear(feature_dim, self.out_dim)  # 模态A -> 键K
        self.v_proj_a = nn.Linear(feature_dim, self.out_dim)  # 模态A -> 值V

        # 输出投影层
        self.out_proj_a = nn.Linear(self.out_dim, self.out_dim)
        self.out_proj_b = nn.Linear(self.out_dim, self.out_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, modal_a, modal_b):
        """
        前向传播

        参数:
            modal_a: 模态A特征，形状为 (batch_size, feature_dim)
            modal_b: 模态B特征，形状为 (batch_size, feature_dim)

        返回:
            fused_feature: 融合后的特征，形状为 (batch_size, out_dim)
            attn_weights_a: 模态A关注模态B的注意力权重
            attn_weights_b: 模态B关注模态A的注意力权重
        """
        batch_size = modal_a.size(0)

        # 为了适应注意力机制，增加一个虚拟的序列维度 (长度为1)
        # 形状变为 (batch_size, 1, feature_dim)
        a = modal_a.unsqueeze(1)
        b = modal_b.unsqueeze(1)

        # ------------------------------
        # 第一部分：模态A关注模态B
        # ------------------------------
        q_a = self.q_proj_a(a)  # (batch_size, 1, out_dim)
        k_b = self.k_proj_b(b)  # (batch_size, 1, out_dim)
        v_b = self.v_proj_b(b)  # (batch_size, 1, out_dim)

        # 分割成多个注意力头
        q_a = q_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)
        k_b = k_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)
        v_b = v_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)

        # 计算注意力分数
        attn_scores_a = torch.matmul(q_a, k_b.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)
        attn_weights_a = F.softmax(attn_scores_a, dim=-1)  # (B, H, 1, 1)
        attn_weights_a = self.dropout(attn_weights_a)

        # 应用注意力
        output_a = torch.matmul(attn_weights_a, v_b)  # (B, H, 1, D/H)
        output_a = output_a.transpose(1, 2).contiguous().view(batch_size, -1, self.out_dim)  # (B, 1, D)
        output_a = self.out_proj_a(output_a).squeeze(1)  # (B, D)

        # ------------------------------
        # 第二部分：模态B关注模态A
        # ------------------------------
        q_b = self.q_proj_b(b)  # (batch_size, 1, out_dim)
        k_a = self.k_proj_a(a)  # (batch_size, 1, out_dim)
        v_a = self.v_proj_a(a)  # (batch_size, 1, out_dim)

        # 分割成多个注意力头
        q_b = q_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)
        k_a = k_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)
        v_a = v_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)

        # 计算注意力分数
        attn_scores_b = torch.matmul(q_b, k_a.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)
        attn_weights_b = F.softmax(attn_scores_b, dim=-1)  # (B, H, 1, 1)
        attn_weights_b = self.dropout(attn_weights_b)

        # 应用注意力
        output_b = torch.matmul(attn_weights_b, v_a)  # (B, H, 1, D/H)
        output_b = output_b.transpose(1, 2).contiguous().view(batch_size, -1, self.out_dim)  # (B, 1, D)
        output_b = self.out_proj_b(output_b).squeeze(1)  # (B, D)

        # ------------------------------
        # 融合两个方向的注意力结果
        # ------------------------------
        # 可以使用多种融合方式，这里使用简单的拼接+线性变换
        #fused_feature = torch.cat([output_a, output_b, modal_a, modal_b], dim=1)
        fused_feature = torch.cat([output_a, output_b], dim=1)

        return fused_feature, (attn_weights_a, attn_weights_b)

# 完整的多模态回归模型
class MultimodalRegressionModel(nn.Module):
    """使用交叉注意力融合B×N形状的多模态特征并进行回归预测"""

    def __init__(self, feature_dim, num_heads=4, dropout=0.2):
        super().__init__()
        # 交叉注意力融合模块
        self.cross_attention = CrossAttentionForBN(
            feature_dim=feature_dim,
            num_heads=num_heads,
            out_dim=feature_dim,
            dropout=dropout
        )

    def forward(self, modal1, modal2):
        """
        参数:
            modal1: 第一模态特征，形状为 (batch_size, feature_dim)
            modal2: 第二模态特征，形状为 (batch_size, feature_dim)
        """
        # 交叉注意力融合
        fused_feature, attn_weights = self.cross_attention(modal1, modal2)

        return fused_feature, attn_weights

'''
# 使用示例
if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)

    # 模拟B×N形状的多模态特征
    batch_size = 8
    feature_dim = 64  # N=64

    # 模态1特征: (batch_size, feature_dim)
    modal1 = torch.randn(batch_size, feature_dim)
    # 模态2特征: (batch_size, feature_dim)
    modal2 = torch.randn(batch_size, feature_dim)

    # 初始化模型
    model = MultimodalRegressionModel(
        feature_dim=feature_dim,
        num_heads=4
    )

    # 前向传播
    output, attn_weights = model(modal1, modal2)

    print(f"模态1输入形状: {modal1.shape}")
    print(f"模态2输入形状: {modal2.shape}")
    print(f"回归输出形状: {output.shape}")
    print(f"模态1关注模态2的注意力权重形状: {attn_weights[0].shape}")
    print(f"模态2关注模态1的注意力权重形状: {attn_weights[1].shape}")
'''