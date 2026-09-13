import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionForBN_MCAFM_FR(nn.Module):
    """
    适用于B×N形状特征的交叉注意力模块（支持不同输入维度）
    输入: 两个模态的特征，形状分别为 (batch_size, feature_dim_a) 和 (batch_size, feature_dim_b)
    输出: 融合后的特征，形状为 (batch_size, out_dim * 2) （因拼接output_a和output_b，维度翻倍）
    """

    def __init__(self, feature_dim_a, feature_dim_b, out_dim, num_heads=4,  dropout=0.1):
        """
        参数:
            feature_dim_a: 模态A输入特征的维度
            feature_dim_b: 模态B输入特征的维度
            num_heads: 注意力头的数量
            out_dim: 单个模态注意力输出的维度，默认与模态A输入维度相同
            dropout: dropout概率
        """
        super().__init__()
        # 单个模态注意力的输出维度，默认沿用模态A的输入维度
        #self.out_dim = out_dim if out_dim is not None else feature_dim_a+feature_dim_b
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = self.out_dim // num_heads

        # 确保输出维度可以被注意力头数量整除（核心约束不变）
        assert self.head_dim * num_heads == self.out_dim, "输出维度必须是注意力头数量的整数倍"

        # ------------------------------
        # 模态A作为查询，模态B作为键和值的投影层（适配不同输入维度）
        # ------------------------------
        self.q_proj_a = nn.Linear(feature_dim_a, self.out_dim)  # A→Q：输入维度=feature_dim_a
        self.k_proj_b = nn.Linear(feature_dim_b, self.out_dim)  # B→K：输入维度=feature_dim_b
        self.v_proj_b = nn.Linear(feature_dim_b, self.out_dim)  # B→V：输入维度=feature_dim_b
        self.out_proj_a = nn.Linear(self.out_dim, self.out_dim)  # A侧输出投影

        # ------------------------------
        # 模态B作为查询，模态A作为键和值的投影层（适配不同输入维度）
        # ------------------------------
        self.q_proj_b = nn.Linear(feature_dim_b, self.out_dim)  # B→Q：输入维度=feature_dim_b
        self.k_proj_a = nn.Linear(feature_dim_a, self.out_dim)  # A→K：输入维度=feature_dim_a
        self.v_proj_a = nn.Linear(feature_dim_a, self.out_dim)  # A→V：输入维度=feature_dim_a
        self.out_proj_b = nn.Linear(self.out_dim, self.out_dim)  # B侧输出投影

        # 其他层（与原逻辑一致）
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # 注意力分数缩放因子

    def forward(self, modal_a, modal_b):
        """
        前向传播
        参数:
            modal_a: 模态A特征，形状为 (batch_size, feature_dim_a)
            modal_b: 模态B特征，形状为 (batch_size, feature_dim_b)
        返回:
            fused_feature: 融合后的特征，形状为 (batch_size, out_dim * 2)
            attn_weights_a: 模态A关注模态B的注意力权重，形状为 (batch_size, num_heads, 1, 1)
            attn_weights_b: 模态B关注模态A的注意力权重，形状为 (batch_size, num_heads, 1, 1)
        """
        batch_size = modal_a.size(0)
        # 增加虚拟序列维度（长度=1），适配注意力机制的序列输入格式
        a = modal_a.unsqueeze(1)  # 形状: (batch_size, 1, feature_dim_a)
        b = modal_b.unsqueeze(1)  # 形状: (batch_size, 1, feature_dim_b)

        # ------------------------------
        # 第一部分：模态A关注模态B（逻辑完全不变）
        # ------------------------------
        q_a = self.q_proj_a(a)  # (batch_size, 1, out_dim)
        k_b = self.k_proj_b(b)  # (batch_size, 1, out_dim)
        v_b = self.v_proj_b(b)  # (batch_size, 1, out_dim)

        # 分割为多个注意力头（维度转换：B×1×D → B×H×1×(D/H)）
        q_a = q_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_b = k_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_b = v_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数与权重
        attn_scores_a = torch.matmul(q_a, k_b.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)
        attn_weights_a = F.softmax(attn_scores_a, dim=-1)
        attn_weights_a = self.dropout(attn_weights_a)

        # 应用注意力并恢复维度
        output_a = torch.matmul(attn_weights_a, v_b)  # (B, H, 1, D/H)
        output_a = output_a.transpose(1, 2).contiguous().view(batch_size, -1, self.out_dim)  # (B, 1, D)
        output_a = self.out_proj_a(output_a).squeeze(1)  # (B, D)

        # ------------------------------
        # 第二部分：模态B关注模态A（逻辑完全不变）
        # ------------------------------
        q_b = self.q_proj_b(b)  # (batch_size, 1, out_dim)
        k_a = self.k_proj_a(a)  # (batch_size, 1, out_dim)
        v_a = self.v_proj_a(a)  # (batch_size, 1, out_dim)

        # 分割为多个注意力头
        q_b = q_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = k_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = v_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数与权重
        attn_scores_b = torch.matmul(q_b, k_a.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)
        attn_weights_b = F.softmax(attn_scores_b, dim=-1)
        attn_weights_b = self.dropout(attn_weights_b)

        # 应用注意力并恢复维度
        output_b = torch.matmul(attn_weights_b, v_a)  # (B, H, 1, D/H)
        output_b = output_b.transpose(1, 2).contiguous().view(batch_size, -1, self.out_dim)  # (B, 1, D)
        output_b = self.out_proj_b(output_b).squeeze(1)  # (B, D)

        # ------------------------------
        # 融合两个方向的结果（与原逻辑一致，拼接后维度为 2*out_dim）
        # ------------------------------
        fused_feature = torch.cat([output_a, output_b], dim=1)

        return fused_feature, (attn_weights_a, attn_weights_b)