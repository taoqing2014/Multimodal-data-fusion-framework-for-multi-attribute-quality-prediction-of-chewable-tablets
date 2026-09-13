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
    mat_data = scipy.io.loadmat('iMCAFM_FR/ys.mat')  # 替换为你的.mat文件路径
    cal_idx = mat_data['cal_idx']
    test_idx = mat_data['test_idx']

    ys = mat_data['ys']
    y = ys[:, ylabel_idx]
    return cal_idx, test_idx, y

def load_fs(ylabel_idx=0):

    nir_file_name = f'iMCAFM_FR/nirFS_{ylabel_idx}.mat'
    mat_data = scipy.io.loadmat(nir_file_name)
    nir_fs = mat_data['nir_fs']
    hsi_file_name = f'iMCAFM_FR/hsiFS_{ylabel_idx}.mat'
    mat_data = scipy.io.loadmat(hsi_file_name)
    hsi_fs = mat_data['hsi_fs']

    return nir_fs, hsi_fs

def load_fs_fs(ylabel_idx=0):

    cal_idx, test_idx, y = load_ys(ylabel_idx)
    nir_fs, hsi_fs = load_fs(ylabel_idx)
    return cal_idx[0], test_idx[0], y, nir_fs, hsi_fs



class MulHeadAttention(nn.Module):
    """交叉注意力模块：使用一组特征作为查询，另一组特征作为键和值"""

    def __init__(self, dim_q: int, dim_kv: int, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"

        # 投影层：将输入特征映射到d_model维度
        self.q_proj = nn.Linear(dim_q, d_model)
        self.k_proj = nn.Linear(dim_kv, d_model)
        self.v_proj = nn.Linear(dim_kv, d_model)

        # 输出投影层
        self.out_proj = nn.Linear(d_model, d_model)

        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        Args:
            query: 查询特征，形状为 [batch_size, seq_len_q, dim_q]
            key_value: 键值特征，形状为 [batch_size, seq_len_kv, dim_kv]

        Returns:
            注意力输出，形状为 [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        # 投影到d_model维度
        q = self.q_proj(query)  # [batch_size, seq_len_q, d_model]
        k = self.k_proj(key_value)  # [batch_size, seq_len_kv, d_model]
        v = self.v_proj(key_value)  # [batch_size, seq_len_kv, d_model]

        # 多头注意力：分割成多个头
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1,
                                                                                   2)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # [batch_size, num_heads, seq_len_kv, head_dim]
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # [batch_size, num_heads, seq_len_kv, head_dim]

        # 计算注意力分数
        attn_scores = torch.matmul(q,
                                   k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len_q, seq_len_kv]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_kv]

        # 应用注意力权重
        output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len_q, head_dim]

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q,
                                                          self.d_model)  # [batch_size, seq_len_q, d_model]

        # 输出投影
        output = self.out_proj(output)  # [batch_size, seq_len_q, d_model]

        return output


class FeatureFusionRegressor(nn.Module):
    """特征融合回归模型：使用交叉注意力融合高光谱和近红外光谱特征"""

    def __init__(self, hsi_dim: int, nir_dim: int, d_model: int = 128, num_heads: int = 4, hidden_dim: int = 256,
                 output_dim: int = 1):
        super().__init__()
        # 将特征转换为序列形式（增加一个序列长度维度）
        # 这里将每个特征视为序列长度为1的序列，也可以根据需要调整

        # 交叉注意力层
        self.hsi_to_nir_attn = MulHeadAttention(hsi_dim, nir_dim, d_model, num_heads)
        self.nir_to_hsi_attn = MulHeadAttention(nir_dim, hsi_dim, d_model, num_heads)

        self.hsi_proj = nn.Linear(hsi_dim, d_model)
        self.nir_proj = nn.Linear(nir_dim, d_model)

        # 前馈网络
        self.ffn_hsi = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.ffn_nir = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # 回归头
        '''
        self.regression_head = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        '''
        self.regression_head = nn.Sequential(
            nn.Linear( d_model * 2, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, hsi_features: Tensor, nir_features: Tensor) -> Tensor:
        """
        Args:
            hsi_features: 高光谱特征，形状为 [batch_size, hsi_dim]
            nir_features: 近红外光谱特征，形状为 [batch_size, nir_dim]

        Returns:
            回归预测结果，形状为 [batch_size, output_dim]
        """
        # 增加序列维度，使特征符合注意力输入格式 [batch_size, seq_len, dim]
        # 这里将每个样本视为长度为1的序列
        #hsi_seq = hsi_features.unsqueeze(1)  # [batch_size, 1, hsi_dim]
        #nir_seq = nir_features.unsqueeze(1)  # [batch_size, 1, nir_dim]
        hsi_seq = hsi_features
        nir_seq = nir_features

        # 交叉注意力：高光谱特征关注近红外特征
        hsi_attn = self.hsi_to_nir_attn(hsi_seq, nir_seq)  # [batch_size, 1, d_model]
        hsi_output = self.norm1(hsi_attn + self.hsi_proj(hsi_seq) )  # 残差连接 + 层归一化
        hsi_output = self.norm2(hsi_output + self.ffn_hsi(hsi_output))  # 前馈网络 + 残差连接 + 层归一化

        # 交叉注意力：近红外特征关注高光谱特征
        nir_attn = self.nir_to_hsi_attn(nir_seq, hsi_seq)  # [batch_size, 1, d_model]
        nir_output = self.norm3(nir_attn + self.nir_proj(nir_seq) )  # 残差连接 + 层归一化
        nir_output = self.norm4(nir_output + self.ffn_nir(nir_output))  # 前馈网络 + 残差连接 + 层归一化

        # 移除序列维度
        hsi_final = hsi_output.squeeze(1)  # [batch_size, d_model]
        nir_final = nir_output.squeeze(1)  # [batch_size, d_model]

        # 融合特征
        fused_features = torch.cat([hsi_final, nir_final], dim=1)  # [batch_size, 2*d_model]

        # 回归预测
        output = self.regression_head(fused_features)  # [batch_size, output_dim]

        return output

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


#set_seed()

class NIR_HSI_Dataset(Dataset):
    def __init__(self, nir_fs, hsi_fs, labels, yscaler=None):
        self.nir_fs = nir_fs
        self.hsi_fs = hsi_fs  # 形状为[N, C, H, W]的数组
        self.labels = labels  # 形状为[N, 1]的一维数组

        self.yscaler = yscaler

    def __len__(self):
        return len(self.hsi_fs)

    def __getitem__(self, idx):
        nir = self.nir_fs[idx].reshape(1, -1).astype(np.float32)
        hsi = self.hsi_fs[idx].reshape(1, -1).astype(np.float32)
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
        #min_delta = avg_val_loss/2

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


def train_models(feature_dims=[32,48,64,128,256],yidxs=[0,1,2,3]):

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001

    #cal_idx, test_idx, nir_fs, hsi_fs, ys = load_nir_Hsi_ys()

    
    #from sklearn.preprocessing import MinMaxScaler
    #scaler_ys = MinMaxScaler()


    #cal_idx = (cal_idx-1)[0]
    #test_idx = (test_idx-1)[0]
    import numpy as np


    #for ylabel_idx in [0,1,2,3,4]:
    #    for feature_dim in [32, 48, 64, 128]:
    for ylabel_idx in yidxs:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        for feature_dim in feature_dims:
            train_hsi = hsi_fs[cal_idx.tolist()]
            test_hsi = hsi_fs[test_idx.tolist()]

            train_spectral = nir_fs[cal_idx,:]
            test_spectral  = nir_fs[test_idx,:]

            yscaler = MinMaxScaler()
            yscale =  yscaler.fit_transform( (ys).reshape(-1, 1) )

            train_quality = yscale[cal_idx,:]
            test_quality = yscale[test_idx,:]

            hsi_dim = train_hsi.shape[1]        # HSI长度
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
            model = FeatureFusionRegressor(
                    hsi_dim=hsi_dim,
                    nir_dim=nir_dim,
                    d_model=128,
                    num_heads=4,
                    hidden_dim=feature_dim,
                    output_dim=1  # 假设是单输出回归问题
                ).to(device)
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

    #print(f'Test Loss: {avg_test_loss:.4f}, Test RMSE: {avg_test_rmse:.4f}')

    # 计算R²分数
    y_true = np.array(all_true_values).reshape(-1, 1)
    y_pred = np.array(all_predictions).reshape(-1, 1)



    if scaler_y is not None:
        y_true, y_pred = scaler_y.inverse_transform(y_true), scaler_y.inverse_transform(y_pred)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total)
    #print(f'R² Score: {r2_score:.4f}')

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if isShowYs == True:
        return {
            'loss': avg_test_loss,
            'rmse': rmse,
            'r2': r2_score,
            'predictions': y_pred.reshape(1,-1),
            'true_values': y_true.reshape(1,-1)
        }


    return {
        'loss': avg_test_loss,
        'rmse': rmse,
        'r2': r2_score,
        #'predictions': y_pred,
        #'true_values': y_true
    }

def test_models(feature_dims=[32,48,64,128,256],yidxs=[0,1,2,3]):
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


            hsi_dim = train_hsi.shape[1]        # HSI长度
            nir_dim = train_spectral.shape[1]  # 光谱长度

            train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
            test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 创建模型实例
            model = FeatureFusionRegressor(
                    hsi_dim=hsi_dim,
                    nir_dim=nir_dim,
                    d_model=128,
                    num_heads=4,
                    hidden_dim=feature_dim,
                    output_dim=1  # 假设是单输出回归问题
                ).to(device)

            save_path = f'HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

            model.load_state_dict(torch.load(save_path, weights_only=True))

            train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
            test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

            print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')
''''''
def test_best_models():
    batch_size = 32


    # 注意力 +拼接 原始
    #feature_dims = [32,64,64,32,48]
    #for ylabel_idx in [0,1,2,3,4]:

    # 注意力
    feature_dims = [128,64,256,256,64]
    for ylabel_idx in [0,1,2,3,4]:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        feature_dim = feature_dims[ylabel_idx]
        train_hsi = hsi_fs[cal_idx.tolist()]
        test_hsi = hsi_fs[test_idx.tolist()]

        train_spectral = nir_fs[cal_idx, :]
        test_spectral = nir_fs[test_idx, :]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys ).reshape(-1, 1))

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

        model = FeatureFusionRegressor(
            hsi_dim=hsi_dim,
            nir_dim=nir_dim,
            d_model=128,
            num_heads=4,
            hidden_dim=feature_dim,
            output_dim=1  # 假设是单输出回归问题
        ).to(device)

        save_path = f'iMCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))

        train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
        test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

        print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')

        def test_best_models():
            batch_size = 32

            # 注意力 +拼接 原始
            # feature_dims = [32,64,64,32,48]
            # for ylabel_idx in [0,1,2,3,4]:

            # 注意力
            feature_dims = [128, 64, 256, 256, 64]
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

                model = FeatureFusionRegressor(
                    hsi_dim=hsi_dim,
                    nir_dim=nir_dim,
                    d_model=128,
                    num_heads=4,
                    hidden_dim=feature_dim,
                    output_dim=1  # 假设是单输出回归问题
                ).to(device)

                save_path = f'iMCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

                model.load_state_dict(torch.load(save_path, weights_only=True))

                train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
                test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

                print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')

def test_best_models_forMainUI():
    batch_size = 32


    # 注意力 +拼接 原始
    #feature_dims = [32,64,64,32,48]
    #for ylabel_idx in [0,1,2,3,4]:
    ylabels = ['孔隙率', '硬度', '抗张强度', '芸香柚皮苷', '橙皮苷']
    results = {}
    # 注意力
    feature_dims = [128,64,256,256,64]
    for ylabel_idx in [0,1,2,3,4]:
        cal_idx, test_idx, ys, nir_fs, hsi_fs = load_fs_fs(ylabel_idx)
        feature_dim = feature_dims[ylabel_idx]
        train_hsi = hsi_fs[cal_idx.tolist()]
        test_hsi = hsi_fs[test_idx.tolist()]

        train_spectral = nir_fs[cal_idx, :]
        test_spectral = nir_fs[test_idx, :]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys ).reshape(-1, 1))

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

        model = FeatureFusionRegressor(
            hsi_dim=hsi_dim,
            nir_dim=nir_dim,
            d_model=128,
            num_heads=4,
            hidden_dim=feature_dim,
            output_dim=1  # 假设是单输出回归问题
        ).to(device)

        save_path = f'iMCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))

        train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler, isShowYs=True)
        test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler, isShowYs=True)

        #print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')
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
    feature_dims = [128,64,256,256,64]
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

        model = FeatureFusionRegressor(
            hsi_dim=hsi_dim,
            nir_dim=nir_dim,
            d_model=128,
            num_heads=4,
            hidden_dim=feature_dim,
            output_dim=1  # 假设是单输出回归问题
        ).to(device)

        save_path = f'iMCAFM_FR/HSI_NIR_AttentionCAT_models/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

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
        save_prediction_path = f'iMCAFM_FR/HSI_NIR_AttentionCAT_models/ypred_{label}.mat'
        sio.savemat(save_prediction_path, val)

    return results

if __name__ == "__main__":
    #feature_dims = [32, 48, 64, 128, 256]
    #yidxs = [0, 1, 2, 3]

    '''
    feature_dims = [32, 48, 64, 128, 256]
    yidxs = [0,1,2,3,4]
    train_models(feature_dims, yidxs)
    test_models(feature_dims, yidxs)
    #'''
    #test_best_models()
    res = test_best_models_forSavePrediction()
    print(res)
