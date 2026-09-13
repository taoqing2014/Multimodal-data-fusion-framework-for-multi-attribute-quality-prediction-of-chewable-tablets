import code

import scipy.io
from pyexpat import features
from ImportDataset import load_hsi_imgs
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from MulScaleResnetSpectral import SpectralFeatureExtractor
from MulScaleResnetSpatical import SpatialFeatureExtractor
from FusionSpaticalSpectral import SpectralSpatialRegressionNetwork, SpectralSpatialRegressionNetworkGetFs

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # 形状为[N, C, H, W]的数组
        self.labels = labels  # 形状为[N, 1]的一维数组
        self.transform = transform  # 图像增强/预处理函数

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]


        if self.transform:
            img = self.transform(img)
        return img, label

def predictor_model(target_dim = 32 ):
    # 目标通道数
    channels = 380 # 高光谱通道数 400-1000 128  |  1000-2500 380
    # 初始化光谱和空间特征提取器
    spectral_extractor = SpectralFeatureExtractor(input_channels=channels, feature_dim=target_dim)
    spatial_extractor = SpatialFeatureExtractor(input_channels=channels, feature_dim=target_dim)

    # 初始化融合回归网络
    model = SpectralSpatialRegressionNetwork(
        spectral_extractor=spectral_extractor,
        spatial_extractor=spatial_extractor,
        feature_dim= target_dim,
        output_dim=1
    )
    return model

def predictor_model_getFs(target_dim = 32 ):
    # 目标通道数
    channels = 380 # 高光谱通道数 400-1000 128  |  1000-2500 380
    # 初始化光谱和空间特征提取器
    spectral_extractor = SpectralFeatureExtractor(input_channels=channels, feature_dim=target_dim)
    spatial_extractor = SpatialFeatureExtractor(input_channels=channels, feature_dim=target_dim)

    # 初始化融合回归网络
    model = SpectralSpatialRegressionNetworkGetFs(
        spectral_extractor=spectral_extractor,
        spatial_extractor=spatial_extractor,
        feature_dim= target_dim,
        output_dim=1
    )
    return model
    

def load_Dataset(ylabel_name = 'zmd'):
    Xs_hsi_images = load_hsi_imgs()
    import scipy.io
    mat_data = scipy.io.loadmat('ys.mat')  # 返回字典，键为变量名，值为数据
    #print(data.keys())  # 查看变量名
    ys_raw = mat_data[ylabel_name]  # 获取具体数据（NumPy数组形式） # water_content, D50, C1, C2, repose_angle

    scaler_ys = MinMaxScaler(feature_range=(0, 1))
    # 拟合数据并进行归一化
    ys = scaler_ys.fit_transform(ys_raw)

    cal_idx = mat_data['cal_idx']
    test_idx = mat_data['test_idx']
    cal_idx = cal_idx.tolist()[0]
    test_idx = test_idx.tolist()[0]

    train_images = Xs_hsi_images[cal_idx]
    train_labels = ys[cal_idx]
    test_images = Xs_hsi_images[test_idx]
    test_labels = ys[test_idx]

    #'''
    # 划分数据集
    train_dataset = ImageDataset(train_images, train_labels )
    test_dataset = ImageDataset(test_images, test_labels)

    # 数据加载器（支持批量读取和多线程）
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    #'''
    #mrmfn = MRMFN_HSI(ys.shape[1])
    #mrmfn(Xs_hsi_images)

    return train_loader, test_loader, scaler_ys

# 训练函数 - 无验证集版本
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型（无验证集）

    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备 (cuda或cpu)

    Returns:
        训练好的模型和训练历史
    """
    # 记录训练历史
    history = {'train_loss': []}

    # 模型移至设备
    model.to(device)

    # 早停相关变量
    ''''''
    best_train_loss = float('inf')
    counter = 0
    best_model_weights = None
    patience = 30
    min_delta = 1e-3


    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            # 数据移至设备
            images = images.to(device).float()
            targets = targets.to(device).float().view(-1, 1)  # 确保目标形状正确

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累积损失
            running_loss += loss.item() * images.size(0)

        # 计算平均训练损失
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # 打印 epoch 信息
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print('-' * 50)


        # 早停逻辑（基于训练损失）
        if epoch_train_loss < best_train_loss - min_delta:
            best_train_loss = epoch_train_loss
            best_model_weights = model.state_dict()  # 保存最佳模型权重
            counter = 0  # 重置计数器
        else:
            counter += 1
            if counter >= patience :
                print(f"早停触发！在第 {epoch + 1} 轮停止训练\t best_loss:{best_train_loss}")
                break


        # 加载最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        '''
        if epoch_train_loss < 0.009:
            print(f"早停触发！在第 {epoch + 1} 轮停止训练")
            break
        '''
    return model, history

# 测试函数
def test_model(model, test_loader, criterion, device, scale_ys):
    """
    在测试集上评估模型

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备 (cuda或cpu)

    Returns:
        测试集上的评估指标
    """
    model.eval()  # 设置为评估模式
    test_running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device).float()
            targets = targets.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            test_running_loss += loss.item() * images.size(0)

            # 收集预测结果和目标值
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算测试指标
    all_targets = scale_ys.inverse_transform(all_targets)
    all_preds = scale_ys.inverse_transform(all_preds)

    test_loss = test_running_loss / len(test_loader.dataset)

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    # 打印测试结果
    '''
    print('Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')
    print(f'Test R2 Score: {r2:.4f}')
    '''

    return {
        #'loss': test_loss,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        #'predictions': all_preds.flatten(),
        #'targets': all_targets.flatten()
    }

def train_models():
    # 配置参数
    learning_rate = 0.001
    num_epochs = 200

    ylabels = [ 'c2']#
    target_dims = [32,48,64,128]
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    for target_dim in target_dims:
        for ylabel in ylabels:
            # 加载数据 + 划分数据集（train + test）
            train_loader, test_loader, scale_ys = load_Dataset(ylabel_name=ylabel)


            # 初始化模型、损失函数和优化器
            # 如果你已有自己的model，可以替换这里
            model = predictor_model( target_dim )
            #torch.save(model.state_dict(), 'model_parameters.pth')

            # 回归任务常用的损失函数：均方误差
            #criterion =  MixedLoss(mae_weight=0.7, mse_weight=0.3)      # nn.MSELoss()
            criterion = nn.MSELoss()

            # 优化器
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            # 训练模型
            print(f'*******************   {ylabel}   ******************** ')
            print("开始训练...")
            trained_model, history = train_model(
                model, train_loader, criterion, optimizer, num_epochs, device
            )
            '''
            # 在测试集上评估
            print("\n开始测试...")
            test_results = test_model(trained_model, test_loader, criterion, device, scale_ys)
            print(test_results)
            '''
            sp = f'./val none/Dualbranch_fs{target_dim}_{ylabel}.pth'
            # 保存模型
            torch.save(trained_model, sp)
            print(f"\n模型已保存为 {sp}")

def test_models(bath_path = f'./val none/'):
    ylabels = ['c2'] #'zmd','kxl','yd','kzqd', 'c1']#,
    target_dims = [32,48,64,128]
    for ylabel in ylabels:
        for target_dim in target_dims:

            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'使用设备: {device}')
            #for ylabel in ylabels:
                # 加载数据 + 划分数据集（train + test）

            train_loader, test_loader, scale_ys = load_Dataset(ylabel_name= ylabel)

            #criterion =  MixedLoss(mae_weight=0.7, mse_weight=0.3)    #nn.MSELoss()
            criterion = nn.MSELoss()

            #trained_model = MRMFN_HSI( output_dim=1 )
            #trained_model.to(device)
            #trained_model.load(f'Dualbranch_{ylabel}.pth')
            sp = f'{bath_path}Dualbranch_fs{target_dim}_{ylabel}.pth'
            trained_model = torch.load(sp, weights_only=False)#./3分支3模块-2/
            trained_model.eval()

            train_results = test_model(trained_model, train_loader, criterion, device, scale_ys)
            print(f'************{bath_path}****{ylabel} target dim{target_dim}************')
            print('train set\n',train_results)
            test_results = test_model(trained_model, test_loader, criterion, device, scale_ys)
            print('test set\n',test_results)

from sklearn.model_selection import train_test_split  # 新增导入

def load_Dataset_val(ylabel_name = 'water_content', val_ratio=0.4):
    Xs_hsi_images = load_hsi_imgs()
    import scipy.io
    mat_data = scipy.io.loadmat('ys.mat')
    ys_raw = mat_data[ylabel_name]

    scaler_ys = MinMaxScaler(feature_range=(0, 1))
    ys = scaler_ys.fit_transform(ys_raw)

    cal_idx = mat_data['cal_idx']
    test_idx = mat_data['test_idx']
    cal_idx = cal_idx.tolist()[0]
    test_idx = test_idx.tolist()[0]

    # 校正集数据（需要进一步拆分为训练集和验证集）
    cal_images = Xs_hsi_images[cal_idx]
    cal_labels = ys[cal_idx]

    # 按2:1比例拆分校正集为训练集和验证集（随机种子确保可复现）
    train_images, val_images, train_labels, val_labels = train_test_split(
        cal_images, cal_labels,
        test_size= val_ratio,  # 验证集占0.2 ，训练集占0.8
        random_state=42  # 固定随机种子，保证拆分结果一致
    )

    # 测试集保持不变
    test_images = Xs_hsi_images[test_idx]
    test_labels = ys[test_idx]

    # 创建数据集
    train_dataset = ImageDataset(train_images, train_labels)
    val_dataset = ImageDataset(val_images, val_labels)  # 新增验证集
    test_dataset = ImageDataset(test_images, test_labels)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)  # 新增验证集加载器
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 返回值增加验证集加载器
    return train_loader, val_loader, test_loader, scaler_ys

def train_model_val(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型（带验证集和早停策略）

    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备 (cuda或cpu)

    Returns:
        训练好的模型和训练历史
    """
    # 记录训练历史（增加验证集指标）
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # 模型移至设备
    model.to(device)

    # 早停相关变量（基于验证集损失）
    best_val_loss = float('inf')
    counter = 0
    best_model_weights = None
    patience = 30
    min_delta = 1e-3

    for epoch in range(num_epochs):
        # -------------------
        # 训练阶段
        # -------------------
        model.train()
        train_running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device).float()
            targets = targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * images.size(0)

        # 计算平均训练损失
        epoch_train_loss = train_running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # -------------------
        # 验证阶段
        # -------------------
        model.eval()  # 切换到评估模式
        val_running_loss = 0.0

        with torch.no_grad():  # 关闭梯度计算
            for images, targets in val_loader:
                images = images.to(device).float()
                targets = targets.to(device).float().view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)

        # 计算平均验证损失
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        # 打印 epoch 信息
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        print('-' * 50)

        # 早停逻辑（基于验证集损失）
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict()  # 保存最佳模型权重
            counter = 0  # 重置计数器
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发！在第 {epoch + 1} 轮停止训练\t Best Val Loss: {best_val_loss:.4f}")
                break

    # 加载最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, history

def train_models_val(val_ratio=0.4):
    # 配置参数
    learning_rate = 0.001
    num_epochs = 200

    #ylabels = ['water_content' , 'D50']
    ylabels = ['c2']  # 'zmd',['kxl', 'yd', 'kzqd','c1']#
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    target_dims = [32, 48, 64, 128]

    for target_dim in target_dims:
        for ylabel in ylabels:
            # 加载数据 + 划分数据集（train + test）
            train_loader, val_loader, test_loader, scale_ys = load_Dataset_val(ylabel_name=ylabel, val_ratio=val_ratio)


            # 初始化模型、损失函数和优化器
            # 如果你已有自己的model，可以替换这里
            model = predictor_model( target_dim=target_dim )
            #torch.save(model.state_dict(), 'model_parameters.pth')

            # 回归任务常用的损失函数：均方误差
            #criterion =  MixedLoss(mae_weight=0.7, mse_weight=0.3)      # nn.MSELoss()
            criterion = nn.MSELoss()

            # 优化器
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            # 训练模型
            print(f'*******************   {ylabel}   ******************** ')
            print("开始训练...")
            trained_model, history = train_model_val(
                model, train_loader, val_loader, criterion, optimizer, num_epochs, device
            )
            '''
            # 在测试集上评估
            print("\n开始测试...")
            test_results = test_model(trained_model, test_loader, criterion, device, scale_ys)
            print(test_results)
            '''
            # 保存模型
            sp = f'./val {val_ratio}/Dualbranch_fs{target_dim}_{ylabel}.pth'
            torch.save(trained_model, sp)
            print(f"\n模型已保存为 {sp}'")

def test_best_models(bath_path = f'./best/'):
    ylabels = ['kxl','yd','kzqd', 'c1', 'c2'] #'zmd',
    target_dims = [48,48,48,48,32]
    for idx, ylabel in enumerate(ylabels):
        #for target_dim in target_dims:
        target_dim = target_dims[idx]
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        #for ylabel in ylabels:
            # 加载数据 + 划分数据集（train + test）

        train_loader, test_loader, scale_ys = load_Dataset(ylabel_name= ylabel)

        #criterion =  MixedLoss(mae_weight=0.7, mse_weight=0.3)    #nn.MSELoss()
        criterion = nn.MSELoss()

        #trained_model = MRMFN_HSI( output_dim=1 )
        #trained_model.to(device)
        #trained_model.load(f'Dualbranch_{ylabel}.pth')
        sp = f'{bath_path}Dualbranch_fs{target_dim}_{ylabel}.pth'
        trained_model = torch.load(sp, weights_only=False)#./3分支3模块-2/
        trained_model.eval()

        train_results = test_model(trained_model, train_loader, criterion, device, scale_ys)
        print(f'************{bath_path}****{ylabel} target dim{target_dim}************')
        print('train set\n',train_results)
        test_results = test_model(trained_model, test_loader, criterion, device, scale_ys)
        print('test set\n',test_results)

if __name__ == "__main__":

    code = 2  # 0 测试模型性能  1 获取特征  2 导出最佳模型的预测值
    if code == 0:
        # 构建回归模型
        #train_models()
        #train_models_val(val_ratio=0.4)
        #train_models_val(val_ratio=0.2)
        #train_models_val(val_ratio=0.1)
        #test_models()
        #test_models(f'./val 0.4/')
        #test_models(f'./val 0.2/')
        #test_models(f'./val 0.1/')

        test_best_models()

    elif code == 1:
        # 提取特征
        #'''
        bath_path = f'./best/'
        ylabels = ['kxl', 'yd', 'kzqd', 'c1', 'c2']  # 'zmd',
        target_dims = [48, 48, 48, 48, 32]
        Xs_hsi_images = load_hsi_imgs()
        global_pool = nn.AdaptiveAvgPool2d(1)
        for idx, ylabel in enumerate(ylabels):
            # for target_dim in target_dims:
            target_dim = target_dims[idx]
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'使用设备: {device}')

            sp = f'{bath_path}Dualbranch_fs{target_dim}_{ylabel}.pth'
            trained_model = torch.load(sp, weights_only=False)
            #trained_model.eval()
            model_para = trained_model.state_dict()
            model = predictor_model_getFs(target_dim=target_dim)
            model.load_state_dict(model_para)
            model.eval()
            hsi_fs = model.getFeatures( Xs_hsi_images )

            hsi_fs = global_pool(hsi_fs).view(hsi_fs.size(0), -1)
            hsi_fs = hsi_fs.detach().cpu().numpy()
            scipy.io.savemat(f'hsiFS_{idx}.mat', {'hsi_fs': hsi_fs})

    elif code == 2:
        # 获取预测结果

        #name_models = ['Dualbranch_kxl-val-04.pth', 'Dualbranch_yd-val-none.pth', 'Dualbranch_kzqd-val-01.pth']
        base_path = './best/'
        name_models = ['Dualbranch_fs48_kxl.pth', 'Dualbranch_fs48_yd.pth', 'Dualbranch_fs48_kzqd.pth', 'Dualbranch_fs48_c1.pth' , 'Dualbranch_fs32_c2.pth']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        ylabels = ['kxl', 'yd', 'kzqd', 'c1', 'c2']  # 'zmd',
        for idx, ylabel in enumerate(ylabels):
            # 加载数据 + 划分数据集（train + test）
            # train_loader, test_loader, scale_ys = load_Dataset(ylabel_name=ylabel)
            trained_model = torch.load(base_path+name_models[idx])  # ./3分支3模块-2/
            trained_model.eval()
            train_loader, test_loader, scale_ys = load_Dataset(ylabel_name=ylabel)
            images = load_hsi_imgs()
            images = images.to(device).float()
            ypred = trained_model(images)
            #torch.save(trained_model, f'getfs-{ylabel}.pth')
            numpy_ypred = ypred.detach().cpu().numpy()
            inv_ypred = scale_ys.inverse_transform(numpy_ypred)
            print(numpy_ypred.shape)
            scipy.io.savemat(f'ypred_{ylabel}.mat', {'ypred_hsi_DSSAN': inv_ypred})
