import code

import scipy.io

from ImportDataset import load_hsi_imgs
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split

from MulScaleResnetSpectral import SpectralFeatureExtractor
from MulScaleResnetSpatical import SpatialFeatureExtractor
from FusionSpaticalSpectral import HSI_SpectralSpatial_Feature_Extractor

from NIR_CNN import NIR_CNN_FeatureExtractor
from ImportDataset import load_nir_Hsi_ys

import random

from AttentionFusion_HSI_NIR import CrossAttentionForBN
#from CAFusion_HSI_NIR import AdaptiveFusion1D

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
    def __init__(self, nirs, images, labels, transform=None, yscaler=None):
        self.nirs = nirs
        self.images = images  # 形状为[N, C, H, W]的数组
        self.labels = labels  # 形状为[N, 1]的一维数组
        self.transform = transform  # 图像增强/预处理函数
        self.yscaler = yscaler

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #nir = self.nir
        nir = self.nirs[idx].reshape(1, -1).astype(np.float32)
        img = self.images[idx]#.astype(np.float32)
        label = self.labels[idx].astype(np.float32)

        if self.transform:
            img = self.transform(img)
        return nir, img, label


def HSI_fs(target_dim = 32):
    channels = 380  # 高光谱通道数 400-1000 128  |  1000-2500 380
      # 目标通道数
    # 初始化光谱和空间特征提取器
    spectral_extractor = SpectralFeatureExtractor(input_channels=channels, feature_dim=target_dim)
    spatial_extractor = SpatialFeatureExtractor(input_channels=channels, feature_dim=target_dim)
    hsi_fs_model = HSI_SpectralSpatial_Feature_Extractor(spectral_extractor=spectral_extractor, spatial_extractor=spatial_extractor, feature_dim=target_dim)
    # HSI 提取后特征维度 B*C
    return hsi_fs_model

class HSI_NIR_Regression_MFN_MCAFM(nn.Module):
    """完整的光谱-空间特征融合回归网络"""

    def __init__(self, feature_dim=32, output_dim=1):
        super(HSI_NIR_Regression_MFN_MCAFM, self).__init__()
        self.hsi_extractor = HSI_fs(target_dim=feature_dim)  # HSI特征提取器
        self.nir_extractor = NIR_CNN_FeatureExtractor(target_dim=feature_dim)  # NIR特征提取器
        self.feature_dim = feature_dim
        '''
        # 交叉注意力融合模块
        self.cross_attention = CrossAttentionForBN(
            feature_dim=feature_dim,
            num_heads=4,
            out_dim=feature_dim,
        )
        '''
        #self.fusion_module = AdaptiveFusion1D(feature_dim= feature_dim)
        self.cross_attention = CrossAttentionForBN(feature_dim=feature_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, output_dim)
        )


    def forward(self, x):
        hsi_fs = self.hsi_extractor( x['hsi'] )
        nir_fs = self.nir_extractor( x['nir'] )
        #fs = torch.cat([hsi_fs, nir_fs], dim=1)
        fs, attn_weights  = self.cross_attention( hsi_fs, nir_fs )
        #fs = self.fusion_module( hsi_fs, nir_fs )

        out = fs.view(fs.size(0), -1)  # 拉直
        out = self.regression_head(out)
        return out

'''
# 测试代码
batch_size = 4  # 批量大小
spectrum_length = 256  # 光谱长度（对应原代码注释的 N=256）
H,W = 20,20
nirs = torch.randn(batch_size, 1, spectrum_length)  # 随机生成模拟数据，形状 (4,1,256)
hsis = torch.randn(batch_size, 380, H, W)

x = {'hsi': hsis, 'nir': nirs}

fs = HSI_NIR_Regression_MFN_MCAFM()
res = fs(x)
'''


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
            outputs = model({'nir':spectrums, 'hsi':img})
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

                outputs = model({'nir':spectrums, 'hsi':img})
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

    cal_idx, test_idx, nir_Xs, hsi_imgs, ys = load_nir_Hsi_ys()

    #from sklearn.preprocessing import MinMaxScaler
    #scaler_ys = MinMaxScaler()


    #cal_idx = (cal_idx-1)[0]
    #test_idx = (test_idx-1)[0]
    import numpy as np


    #for ylabel_idx in [0,1,2,3,4]:
    #    for feature_dim in [32, 48, 64, 128]:
    for ylabel_idx in yidxs:
        for feature_dim in feature_dims:
            train_hsi = hsi_imgs[cal_idx.tolist()]
            test_hsi = hsi_imgs[test_idx.tolist()]

            train_spectral = nir_Xs[cal_idx,:]
            test_spectral  = nir_Xs[test_idx,:]

            yscaler = MinMaxScaler()
            yscale =  yscaler.fit_transform( (ys[:, ylabel_idx]).reshape(-1, 1) )

            train_quality = yscale[cal_idx,:]
            test_quality = yscale[test_idx,:]
            #train_quality = train_quality.reshape(-1,1)
            #test_quality = test_quality.reshape(-1,1)

            #spectral_length = train_spectral.shape[1]  # 光谱长度

            train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)

            # 将训练集按4:1划分为校正集和验证集
            calibration_size = int(0.6 * len(train_dataset))  # 4/5作为校正集
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
            model = HSI_NIR_Regression_MFN_MCAFM(feature_dim=feature_dim ).to(device)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加L2正则化

            # 训练模型（使用校正集和验证集）
            print("\n开始训练模型...")
            model, history = train_model(
                model, calibration_loader, validation_loader, criterion, optimizer, device, num_epochs=num_epochs
            )

            save_path = f'MFN_MCAFM/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'
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

            outputs = model({'nir':spectrums, 'hsi':img})
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
    cal_idx, test_idx, nir_Xs, hsi_imgs, ys = load_nir_Hsi_ys()

    # for ylabel_idx in [0,1,2,3,4]:
    #    for feature_dim in [32, 48, 64, 128]:
    for ylabel_idx in yidxs:
        for feature_dim in feature_dims:
            train_hsi = hsi_imgs[cal_idx.tolist()]
            test_hsi = hsi_imgs[test_idx.tolist()]

            train_spectral = nir_Xs[cal_idx, :]
            test_spectral = nir_Xs[test_idx, :]

            yscaler = MinMaxScaler()
            yscale = yscaler.fit_transform((ys[:, ylabel_idx]).reshape(-1, 1))

            train_quality = yscale[cal_idx, :]
            test_quality = yscale[test_idx, :]
            # train_quality = train_quality.reshape(-1,1)
            # test_quality = test_quality.reshape(-1,1)

            # spectral_length = train_spectral.shape[1]  # 光谱长度

            train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
            test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = HSI_NIR_Regression_MFN_MCAFM(feature_dim=feature_dim).to(device)

            save_path = f'MFN_MCAFM/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

            model.load_state_dict(torch.load(save_path, weights_only=True))

            train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
            test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

            print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')

def test_best_models():
    batch_size = 32
    cal_idx, test_idx, nir_Xs, hsi_imgs, ys = load_nir_Hsi_ys()

    # 注意力 +拼接 原始
    #feature_dims = [32,64,64,32,48]
    #for ylabel_idx in [0,1,2,3,4]:

    # 注意力
    feature_dims = [32,32,48,64,48]
    for ylabel_idx in [0,1,2,3,4]:
        #for feature_dim in [32, 48, 64, 128, 256]:
        feature_dim = feature_dims[ylabel_idx]
        train_hsi = hsi_imgs[cal_idx.tolist()]
        test_hsi = hsi_imgs[test_idx.tolist()]

        train_spectral = nir_Xs[cal_idx, :]
        test_spectral = nir_Xs[test_idx, :]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys[:, ylabel_idx]).reshape(-1, 1))

        train_quality = yscale[cal_idx, :]
        test_quality = yscale[test_idx, :]
        # train_quality = train_quality.reshape(-1,1)
        # test_quality = test_quality.reshape(-1,1)

        # spectral_length = train_spectral.shape[1]  # 光谱长度

        train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
        test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = HSI_NIR_Regression_MFN_MCAFM(feature_dim=feature_dim).to(device)

        save_path = f'MFN_MCAFM/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))

        train_res = test_model(model, train_loader, criterion, device, scaler_y=yscaler)
        test_res = test_model(model, test_loader, criterion, device, scaler_y=yscaler)

        print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')

def test_best_models_forMainUI():
    batch_size = 32
    cal_idx, test_idx, nir_Xs, hsi_imgs, ys = load_nir_Hsi_ys()

    # 注意力 +拼接 原始
    #feature_dims = [32,64,64,32,48]
    #for ylabel_idx in [0,1,2,3,4]:
    ylabels = ['孔隙率','硬度','抗张强度','芸香柚皮苷','橙皮苷']
    results = {}
    # 注意力
    feature_dims = [32,32,48,64,48]
    for ylabel_idx in [0,1,2,3,4]:
        #for feature_dim in [32, 48, 64, 128, 256]:
        feature_dim = feature_dims[ylabel_idx]
        train_hsi = hsi_imgs[cal_idx.tolist()]
        test_hsi = hsi_imgs[test_idx.tolist()]

        train_spectral = nir_Xs[cal_idx, :]
        test_spectral = nir_Xs[test_idx, :]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys[:, ylabel_idx]).reshape(-1, 1))

        train_quality = yscale[cal_idx, :]
        test_quality = yscale[test_idx, :]
        # train_quality = train_quality.reshape(-1,1)
        # test_quality = test_quality.reshape(-1,1)

        # spectral_length = train_spectral.shape[1]  # 光谱长度

        train_dataset = NIR_HSI_Dataset(train_spectral, train_hsi, train_quality, yscaler=yscaler)
        test_dataset = NIR_HSI_Dataset(test_spectral, test_hsi, test_quality, yscaler=yscaler)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = HSI_NIR_Regression_MFN_MCAFM(feature_dim=feature_dim).to(device)

        save_path = f'MFN_MCAFM/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

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
    cal_idx, test_idx, nir_Xs, hsi_imgs, ys = load_nir_Hsi_ys()

    #ylabels = ['孔隙率','硬度','抗张强度','芸香柚皮苷','橙皮苷']
    ylabels = ['kxl', 'yd', 'kzqd', 'c1', 'c2']
    results = {}
    feature_dims = [32,32,48,64,48]
    for ylabel_idx in [0,1,2,3,4]:
        #for feature_dim in [32, 48, 64, 128, 256]:
        feature_dim = feature_dims[ylabel_idx]

        yscaler = MinMaxScaler()
        yscale = yscaler.fit_transform((ys[:, ylabel_idx]).reshape(-1, 1))
        all_spectral, all_hsi, all_quality = nir_Xs, hsi_imgs, yscale

        all_dataset = NIR_HSI_Dataset(all_spectral, all_hsi, all_quality, yscaler=yscaler)
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = HSI_NIR_Regression_MFN_MCAFM(feature_dim=feature_dim).to(device)

        save_path = f'MFN_MCAFM/FeatureDIM_{feature_dim}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.eval()

        all_res = test_model(model, all_loader, criterion, device, scaler_y=yscaler, isShowYs=True)
        #print(f'指标-{ylabel_idx}+ 特征数-{feature_dim} + 训练集：{train_res}\n 测试集：{test_res}')
        result = {}
        ypred= all_res['predictions']
        result['ypred'] = ypred
        results[ylabels[ylabel_idx]] = result
        #print(result)
    #print(results)
    import scipy.io as sio
    for label, val in results.items():
        #ypred = val['ypred']
        save_prediction_path = f'MFN_MCAFM/ypred_{label}.mat'
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
    #res = test_best_models_forMainUI()
    #print(res)
    res = test_best_models_forSavePrediction()
    print(res)
