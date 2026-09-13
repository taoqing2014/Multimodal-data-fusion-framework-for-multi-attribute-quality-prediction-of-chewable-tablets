import code

import scipy.io

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
import torch.nn.functional as F

import random

import scipy.io

def load_ys(ylabel_idx=0):
    mat_data = scipy.io.loadmat('MCAFM_FR/ys.mat')  # 替换为你的.mat文件路径
    cal_idx = mat_data['cal_idx']
    test_idx = mat_data['test_idx']

    ys = mat_data['ys']
    y = ys[:, ylabel_idx]
    return cal_idx, test_idx, y

def load_fs(ylabel_idx=0):

    nir_file_name = f'MCAFM_FR/nirFS_{ylabel_idx}.mat'
    mat_data = scipy.io.loadmat(nir_file_name)
    nir_fs = mat_data['nir_fs']
    hsi_file_name = f'MCAFM_FR/hsiFS_{ylabel_idx}.mat'
    mat_data = scipy.io.loadmat(hsi_file_name)
    hsi_fs = mat_data['hsi_fs']

    return nir_fs, hsi_fs

def load_fs_fs(ylabel_idx=0):

    cal_idx, test_idx, y = load_ys(ylabel_idx)
    nir_fs, hsi_fs = load_fs(ylabel_idx)
    return cal_idx[0], test_idx[0], y, nir_fs, hsi_fs


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


class HSI_NIR_Regression(nn.Module):
    """完整的光谱-空间特征融合回归网络"""

    def __init__(self, hsi_dim: int, nir_dim: int, fusion_feature_dim: int, num_heads: int = 4, output_dim: int = 1):
        super(HSI_NIR_Regression, self).__init__()

        self.cross_attention = CrossAttentionForBN_MCAFM_FR(feature_dim_a=hsi_dim, feature_dim_b=nir_dim,
                                                   out_dim=fusion_feature_dim, num_heads=num_heads)

        self.regression_head = nn.Sequential(
            nn.Linear(fusion_feature_dim * 2, fusion_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_feature_dim // 2, output_dim)
        )

    def forward(self, hsi_fs: Tensor, nir_fs: Tensor) -> Tensor:
        fs, attn_weights = self.cross_attention(hsi_fs, nir_fs)

        out = fs.view(fs.size(0), -1)  # 拉直
        out = self.regression_head(out)
        return out


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# set_seed()

class NIR_HSI_Dataset(Dataset):
    def __init__(self, nir_fs, hsi_fs, labels, yscaler=None):
        self.nir_fs = nir_fs
        self.hsi_fs = hsi_fs  # 形状为[N, C, H, W]的数组
        self.labels = labels  # 形状为[N, 1]的一维数组

        self.yscaler = yscaler

    def __len__(self):
        return len(self.hsi_fs)

    def __getitem__(self, idx):
        # nir = self.nir_fs[idx].reshape(1, -1).astype(np.float32)
        # hsi = self.hsi_fs[idx].reshape(1, -1).astype(np.float32)
        nir = self.nir_fs[idx].astype(np.float32)
        hsi = self.hsi_fs[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)

        return nir, hsi, label


# 带早停策略的训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=200, patience=50, min_delta=0.001):
    """
    训练模型并返回训练历史，包含早停策略
    :param patience: 早停 patience - 多少个epoch性能没有提升就停止
    :param min_delta: 最小改进阈值，小于此值视为没有提升
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': []
    }

    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_running_rmse = 0.0
        train_samples = 0

        for spectrums, img, qualities in train_loader:
            spectrums = spectrums.to(device)
            img = img.to(device)
            qualities = qualities.to(device)

            optimizer.zero_grad()
            outputs = model(img, spectrums)
            loss = criterion(outputs, qualities)
            loss.backward()
            optimizer.step()

            rmse = torch.sqrt(loss)
            train_running_loss += loss.item() * spectrums.size(0)
            train_running_rmse += rmse.item() * spectrums.size(0)
            train_samples += spectrums.size(0)

        avg_train_loss = train_running_loss / train_samples
        avg_train_rmse = train_running_rmse / train_samples

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_rmse = 0.0
        val_samples = 0

        with torch.no_grad():
            for spectrums, img, qualities in val_loader:
                spectrums = spectrums.to(device)
                img = img.to(device)
                qualities = qualities.to(device)

                outputs = model(img, spectrums)
                loss = criterion(outputs, qualities)
                rmse = torch.sqrt(loss)

                val_running_loss += loss.item() * spectrums.size(0)
                val_running_rmse += rmse.item() * spectrums.size(0)
                val_samples += spectrums.size(0)

        avg_val_loss = val_running_loss / val_samples
        avg_val_rmse = val_running_rmse / val_samples

        # 保存历史记录
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(avg_train_rmse)
        history['val_rmse'].append(avg_val_rmse)

        # 打印训练进度
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train RMSE: {avg_train_rmse:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val RMSE: {avg_val_rmse:.4f}')

        # 早停检查
        # 如果当前验证损失比最佳损失好min_delta以上
        # min_delta = avg_val_loss/2

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            counter = 0  # 重置计数器
        else:
            counter += 1  # 增加计数器
            print(f"早停计数器: {counter}/{patience}")
            if counter >= patience:
                print(f"早停触发! 在第{epoch + 1}个epoch停止训练")
                break  # 跳出训练循环

    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model, history


def train_models(feature_dims=[32, 48, 64, 128, 256], yidxs=[0, 1, 2, 3]):
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001

    # cal_idx, test_idx, nir_fs, hsi_fs, ys = load_nir_Hsi_ys()

    # from sklearn.preprocessing import MinMaxScaler
    # scaler_ys = MinMaxScaler()

    # cal_idx = (cal_idx-1)[0]
    # test_idx = (test_idx-1)[0]
    import numpy as np

    # for ylabel_idx in [0,1,2,3,4]:
    #    for feature_dim in [32, 48, 64, 128]:
    for ylabel_idx in yidxs:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        for feature_dim in feature_dims:
            train_hsi = hsi_fs[cal_idx.tolist()]
            test_hsi = hsi_fs[test_idx.tolist()]

            train_spectral = nir_fs[cal_idx, :]
            test_spectral = nir_fs[test_idx, :]

            yscaler = MinMaxScaler()
            yscale = yscaler.fit_transform((ys).reshape(-1, 1))

            train_quality = yscale[cal_idx, :]
            test_quality = yscale[test_idx, :]

            hsi_dim = train_hsi.shape[1]  # HSI长度
            nir_dim = train_spectral.shape[1]  # 光谱长度

            train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)

            # 将训练集按4:1划分为校正集和验证集
            calibration_size = int(0.9 * len(train_dataset))  # 4/5作为校正集
            validation_size = len(train_dataset) - calibration_size  # 1/5作为验证集

            calibration_dataset, validation_dataset = random_split(
                train_dataset, [calibration_size, validation_size]
            )

            # 使用训练集的scaler来标准化测试集（重要！保持数据分布一致）
            test_dataset = NIR_HSI_Dataset(
                test_spectral,
                test_hsi,
                test_quality,
                yscaler=yscaler
            )

            # 创建数据加载器
            calibration_loader = DataLoader(calibration_dataset.dataset, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(validation_dataset.dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 打印数据集大小信息
            print(f"校正集大小: {len(calibration_dataset)}")
            print(f"验证集大小: {len(validation_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")

            # 检查GPU是否可用
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {device}")

            # 创建模型实例
            '''
            model = FeatureFusionRegressor(
                    hsi_dim=hsi_dim,
                    nir_dim=nir_dim,
                    d_model=128,
                    num_heads=4,
                    hidden_dim=feature_dim,
                    output_dim=1  # 假设是单输出回归问题
                ).to(device)
            '''
            model = HSI_NIR_Regression(hsi_dim=hsi_dim, nir_dim=nir_dim, fusion_feature_dim=feature_dim, num_heads=4,
                                       output_dim=1).to(device)
            # 定义损失函数和优化器
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加L2正则化

            # 训练模型（使用校正集和验证集）
            print("\n开始训练模型...")
            model, history = train_model(
                model, calibration_loader, validation_loader, criterion, optimizer, device, num_epochs=num_epochs
            )

            save_path = f'HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'
            # 保存模型
            torch.save(model.state_dict(), save_path)
            print(f"\n模型已保存为 {save_path}")


def test_model(model, test_loader, criterion, device, isShowYs=False, scaler_y=None):
    """测试模型性能"""
    model.eval()
    test_loss = 0.0
    test_rmse = 0.0
    samples = 0

    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for spectrums, img, qualities in test_loader:
            spectrums = spectrums.to(device)
            img = img.to(device)
            qualities = qualities.to(device)

            outputs = model(img, spectrums)
            loss = criterion(outputs, qualities)
            rmse = torch.sqrt(loss)

            test_loss += loss.item() * spectrums.size(0)
            test_rmse += rmse.item() * spectrums.size(0)
            samples += spectrums.size(0)

            # 保存预测值和真实值，用于后续可视化
            all_predictions.extend(outputs.cpu().numpy())
            all_true_values.extend(qualities.cpu().numpy())

    avg_test_loss = test_loss / samples
    avg_test_rmse = test_rmse / samples

    # print(f'Test Loss: {avg_test_loss:.4f}, Test RMSE: {avg_test_rmse:.4f}')

    # 计算R²分数
    y_true = np.array(all_true_values).reshape(-1, 1)
    y_pred = np.array(all_predictions).reshape(-1, 1)

    if scaler_y is not None:
        y_true, y_pred = scaler_y.inverse_transform(y_true), scaler_y.inverse_transform(y_pred)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total)
    # print(f'R² Score: {r2_score:.4f}')

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if isShowYs == True:
        return {
            'loss': avg_test_loss,
            'rmse': rmse,
            'r2': r2_score,
            'predictions': y_pred.reshape(1, -1),
            'true_values': y_true.reshape(1, -1)
        }

    return {
        'loss': avg_test_loss,
        'rmse': rmse,
        'r2': r2_score,
        # 'predictions': y_pred,
        # 'true_values': y_true
    }


def test_models(feature_dims=[32, 48, 64, 128, 256], yidxs=[0, 1, 2, 3]):
    batch_size = 32

    # for ylabel_idx in [0,1,2,3,4]:
    #    for feature_dim in [32, 48, 64, 128]:
    for ylabel_idx in yidxs:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        for feature_dim in feature_dims:
            train_hsi = hsi_fs[cal_idx.tolist()]
            test_hsi = hsi_fs[test_idx.tolist()]

            train_spectral = nir_fs[cal_idx, :]
            test_spectral = nir_fs[test_idx, :]

            yscaler = MinMaxScaler()
            yscale = yscaler.fit_transform((ys).reshape(-1, 1))

            train_quality = yscale[cal_idx, :]
            test_quality = yscale[test_idx, :]

            hsi_dim = train_hsi.shape[1]  # HSI长度
            nir_dim = train_spectral.shape[1]  # 光谱长度

            train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
            test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 创建模型实例
            '''
            model = FeatureFusionRegressor(
                    hsi_dim=hsi_dim,
                    nir_dim=nir_dim,
                    d_model=128,
                    num_heads=4,
                    hidden_dim=feature_dim,
                    output_dim=1  # 假设是单输出回归问题
                ).to(device)
            '''
            model = HSI_NIR_Regression(hsi_dim=hsi_dim, nir_dim=nir_dim, fusion_feature_dim=feature_dim, num_heads=4,
                                       output_dim=1).to(device)
            save_path = f'HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

            model.load_state_dict(torch.load(save_path, weights_only=True))

            train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
            test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

            print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')


''''''


def test_best_models():
    batch_size = 32

    # 注意力 +拼接 原始
    # feature_dims = [32,64,64,32,48]
    # for ylabel_idx in [0,1,2,3,4]:

    # 注意力
    feature_dims = [128, 256, 256, 128, 32]
    for ylabel_idx in [0, 1, 2, 3, 4]:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        feature_dim = feature_dims[ylabel_idx]
        train_hsi = hsi_fs[cal_idx.tolist()]
        test_hsi = hsi_fs[test_idx.tolist()]

        train_spectral = nir_fs[cal_idx, :]
        test_spectral = nir_fs[test_idx, :]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys).reshape(-1, 1))

        train_quality = yscale[cal_idx, :]
        test_quality = yscale[test_idx, :]

        hsi_dim = train_hsi.shape[1]  # HSI长度
        nir_dim = train_spectral.shape[1]  # 光谱长度

        train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
        test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''
        model = FeatureFusionRegressor(
            hsi_dim=hsi_dim,
            nir_dim=nir_dim,
            d_model=128,
            num_heads=4,
            hidden_dim=feature_dim,
            output_dim=1  # 假设是单输出回归问题
        ).to(device)
        '''
        model = HSI_NIR_Regression(hsi_dim=hsi_dim, nir_dim=nir_dim, fusion_feature_dim=feature_dim, num_heads=4,
                                   output_dim=1).to(device)
        save_path = f'MCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))

        train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
        test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

        print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')


def test_best_models_forMainUI():
    batch_size = 32

    # 注意力 +拼接 原始
    # feature_dims = [32,64,64,32,48]
    # for ylabel_idx in [0,1,2,3,4]:
    ylabels = ['孔隙率', '硬度', '抗张强度', '芸香柚皮苷', '橙皮苷']
    results = {}
    # 注意力
    feature_dims = [128, 256, 256, 128, 32]
    for ylabel_idx in [0, 1, 2, 3, 4]:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        feature_dim = feature_dims[ylabel_idx]
        train_hsi = hsi_fs[cal_idx.tolist()]
        test_hsi = hsi_fs[test_idx.tolist()]

        train_spectral = nir_fs[cal_idx, :]
        test_spectral = nir_fs[test_idx, :]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys).reshape(-1, 1))

        train_quality = yscale[cal_idx, :]
        test_quality = yscale[test_idx, :]

        hsi_dim = train_hsi.shape[1]  # HSI长度
        nir_dim = train_spectral.shape[1]  # 光谱长度

        train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
        test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''
        model = FeatureFusionRegressor(
            hsi_dim=hsi_dim,
            nir_dim=nir_dim,
            d_model=128,
            num_heads=4,
            hidden_dim=feature_dim,
            output_dim=1  # 假设是单输出回归问题
        ).to(device)
        '''
        model = HSI_NIR_Regression(hsi_dim=hsi_dim, nir_dim=nir_dim, fusion_feature_dim=feature_dim, num_heads=4,
                                   output_dim=1).to(device)
        save_path = f'MCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))

        train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler, isShowYs=True)
        test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler, isShowYs=True)

        # print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')
        result = {}
        R2c, R2p = round(train_res['r2'], 4), round(test_res['r2'], 4)
        RMSEC, RMSEP = round(train_res['rmse'], 4), round(test_res['rmse'], 4)
        ypred_cal, ypred_test = train_res['predictions'], test_res['predictions']
        ytrue_cal, ytrue_test = train_res['true_values'], test_res['true_values']
        result['R2c'] = R2c
        result['R2p'] = R2p
        result['RMSEC'] = RMSEC
        result['RMSEP'] = RMSEP
        result['ypred_cal'] = ypred_cal
        result['ypred_test'] = ypred_test
        result['ytrue_cal'] = ytrue_cal
        result['ytrue_test'] = ytrue_test

        results[ylabels[ylabel_idx]] = result
        # print(result)

    return results


def test_best_models_forSavePrediction():
    batch_size = 32

    # 注意力 +拼接 原始
    # feature_dims = [32,64,64,32,48]
    # for ylabel_idx in [0,1,2,3,4]:
    #ylabels = ['孔隙率', '硬度', '抗张强度', '芸香柚皮苷', '橙皮苷']
    ylabels = ['kxl', 'yd', 'kzqd', 'c1', 'c2']

    results = {}
    # 注意力
    feature_dims = [128, 256, 256, 128, 32]
    for ylabel_idx in [0, 1, 2, 3, 4]:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        feature_dim = feature_dims[ylabel_idx]

        all_hsi = hsi_fs
        all_spectral = nir_fs


        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys).reshape(-1, 1))

        all_quality = yscale

        hsi_dim = all_hsi.shape[1]  # HSI长度
        nir_dim = all_spectral.shape[1]  # 光谱长度

        all_dataset = NIR_HSI_Dataset(all_spectral, all_hsi, all_quality, yscaler=yscaler)

        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = HSI_NIR_Regression(hsi_dim=hsi_dim, nir_dim=nir_dim, fusion_feature_dim=feature_dim, num_heads=4,
                                   output_dim=1).to(device)
        save_path = f'MCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))

        all_res = test_model(model, all_loader, criterion, device, scaler_y=yscaler, isShowYs=True)

        # print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')
        result = {}
        ypred = all_res['predictions']
        result['ypred'] = ypred

        results[ylabels[ylabel_idx]] = result
        # print(result)
    import scipy.io as sio
    for label, val in results.items():
        # ypred = val['ypred']
        save_prediction_path = f'MCAFM_FR/HSI_NIR_AttentionCAT_models/ypred_{label}.mat'
        sio.savemat(save_prediction_path, val)

    return results

if __name__ == "__main__":
    # feature_dims = [32, 48, 64, 128, 256]
    # yidxs = [0, 1, 2, 3]

    '''
    feature_dims = [32, 48, 64, 128, 256]
    yidxs = [0,1,2,3,4]
    train_models(feature_dims, yidxs)
    test_models(feature_dims, yidxs)
    #'''
    #test_best_models()
    #res = test_best_models_forMainUI()
    #print(res)

    res = test_best_models_forSavePrediction()
    print(res)