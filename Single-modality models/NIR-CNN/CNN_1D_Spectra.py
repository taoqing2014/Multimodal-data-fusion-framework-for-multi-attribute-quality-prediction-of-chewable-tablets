import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import ylabel
#from setuptools.sandbox import save_path
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import random

from torcheval.metrics.functional import  mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

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


# 定义光谱数据集类
class SpectralDataset(Dataset):
    def __init__(self, spectral_data, quality_labels, transform=None, scaler=None):
        """
        初始化数据集
        :param spectral_data: 光谱数据，形状为 [样本数, 光谱长度]
        :param quality_labels: 质量指标标签，形状为 [样本数, 1]
        :param transform: 数据变换函数
        :param scaler: 外部传入的标准化器，用于测试集保持与训练集一致的标准化
        """
        self.spectral_data = spectral_data
        self.quality_labels = quality_labels
        self.transform = transform

        # 数据标准化
        if scaler is None:
            self.scaler = StandardScaler()
            self.spectral_data = self.scaler.fit_transform(self.spectral_data)
        else:
            self.scaler = scaler
            self.spectral_data = self.scaler.transform(self.spectral_data)

    def __len__(self):
        return len(self.spectral_data)

    def __getitem__(self, idx):
        # 获取光谱数据并添加通道维度，变为 [1, 光谱长度]
        spectrum = self.spectral_data[idx].reshape(1, -1).astype(np.float32)
        quality = self.quality_labels[idx].astype(np.float32)

        if self.transform:
            spectrum = self.transform(spectrum)

        return torch.from_numpy(spectrum), torch.from_numpy(quality)


class CNN_ConvUnit1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(CNN_ConvUnit1D, self).__init__()

        # 1d convolution
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # batchnorm
        self.batchnorm1d = nn.BatchNorm1d(in_channels)

        # relu layer
        self.PReLU = nn.PReLU()

        # max pooling
        self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.batchnorm1d(x)
        x = self.conv1d(x)
        x = self.PReLU(x)
        x = self.maxpooling(x)
        return x

class CNN_1D_Spectra(nn.Module):

    def __init__(self, in_dim, fc2_size=64):
        super(CNN_1D_Spectra, self).__init__()
        self.out_dims = 1

        self.conv_1 = CNN_ConvUnit1D(1, 8, kernel_size=3)      # 1*N*1 ->  4*(N-1)*1                               N=  256->255
        self.conv_2 = CNN_ConvUnit1D(8, 16, kernel_size=3)      #  4*(N-1)/2*1 ->  8*(N-1)/2-1*1                        127->126
        self.conv_3 = CNN_ConvUnit1D(16, 32, kernel_size=3)     #                                                       63->62

        #self.conv_out_dim = 192 # NIRs 1557->192  HSI 128->14
        self.conv_out_dim = int( (in_dim//10*10)/8 ) - 1
        DropoutP = 0.25
        #self.fc2_size = 256
        self.FC_12 = nn.Sequential(
            nn.Dropout(p= DropoutP),
            nn.Linear(32 * self.conv_out_dim, fc2_size),
            nn.PReLU(),
            nn.Linear(fc2_size, 1)
        )

    def forward(self, x):

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        #self.conv_out_dim = out.size()[2]
        out = out.view(out.size(0), -1)         # 拉直 维度:16 * self.conv_out_dim

        out = self.FC_12(out)
        return out


class CNN_1D_SpectraGetFs(nn.Module):

    def __init__(self, in_dim, fc2_size=64):
        super(CNN_1D_SpectraGetFs, self).__init__()
        self.out_dims = 1

        self.conv_1 = CNN_ConvUnit1D(1, 8, kernel_size=3)      # 1*N*1 ->  4*(N-1)*1                               N=  256->255
        self.conv_2 = CNN_ConvUnit1D(8, 16, kernel_size=3)      #  4*(N-1)/2*1 ->  8*(N-1)/2-1*1                        127->126
        self.conv_3 = CNN_ConvUnit1D(16, 32, kernel_size=3)     #                                                       63->62

        #self.conv_out_dim = 192 # NIRs 1557->192  HSI 128->14
        self.conv_out_dim = int( (in_dim//10*10)/8 ) - 1
        DropoutP = 0.25
        #self.fc2_size = 256
        self.FC_12 = nn.Sequential(
            nn.Dropout(p= DropoutP),
            nn.Linear(32 * self.conv_out_dim, fc2_size),
            nn.PReLU(),
            nn.Linear(fc2_size, 1)
        )

    def getFeatures(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        return out


    def forward(self, x):

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        #self.conv_out_dim = out.size()[2]
        out = out.view(out.size(0), -1)         # 拉直 维度:16 * self.conv_out_dim

        out = self.FC_12(out)
        return out


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

        for spectrums, qualities in train_loader:
            spectrums = spectrums.to(device)
            qualities = qualities.to(device)

            optimizer.zero_grad()
            outputs = model(spectrums)
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
            for spectrums, qualities in val_loader:
                spectrums = spectrums.to(device)
                qualities = qualities.to(device)

                outputs = model(spectrums)
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


# 测试函数
def test_model(model, test_loader, criterion, device, scaler_y=None, isShowYs=False):
    """测试模型性能"""
    model.eval()
    test_loss = 0.0
    test_rmse = 0.0
    samples = 0

    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for spectrums, qualities in test_loader:
            spectrums = spectrums.to(device)
            qualities = qualities.to(device)

            outputs = model(spectrums)
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
    y_true = np.array(all_true_values)
    y_pred = np.array(all_predictions)

    if scaler_y is not None:
        y_true, y_pred = scaler_y.inverse_transform(y_true), scaler_y.inverse_transform(y_pred)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    #print(f'R² Score: {r2:.4f}')

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if isShowYs == True:
        return {
            'loss': avg_test_loss,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred.reshape(1,-1),
            'true_values': y_true.reshape(1,-1)
        }


    return {
        'loss': avg_test_loss,
        'rmse': rmse,
        'r2': r2,
        #'predictions': y_pred,
        #'true_values': y_true
    }

def main(xlabel = 'nir'):

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001

    from load_datas import load_nir_ys

    cal_idx, test_idx, nir_Xs, ys = load_nir_ys()

    #from sklearn.preprocessing import MinMaxScaler
    #scaler_ys = MinMaxScaler()


    #cal_idx = (cal_idx-1)[0]
    #test_idx = (test_idx-1)[0]
    import numpy as np


    for ylabel_idx in [4]:#[0,1,2,3,4]:
        for fc2_size in [32, 48,64,128,256]:


            train_spectral = nir_Xs[cal_idx,:]
            test_spectral  = nir_Xs[test_idx,:]

            ys_scaler = MinMaxScaler()
            ys_scale =  ys_scaler.fit_transform( (ys[:, ylabel_idx]).reshape(-1,1) )

            train_quality = ys_scale[cal_idx,:] #ys[cal_idx,:]
            test_quality =  ys_scale[test_idx,:] #ys[test_idx,:]
            #train_quality = train_quality[:, ylabel_idx].reshape(-1,1)
            #test_quality = test_quality[:, ylabel_idx].reshape(-1,1)


            spectral_length = train_spectral.shape[1]  # 光谱长度

            train_dataset = SpectralDataset(train_spectral, train_quality)

            # 将训练集按4:1划分为校正集和验证集
            calibration_size = int(0.6 * len(train_dataset))  # 4/5作为校正集
            validation_size = len(train_dataset) - calibration_size  # 1/5作为验证集

            calibration_dataset, validation_dataset = random_split(
                train_dataset, [calibration_size, validation_size]
            )

            # 使用训练集的scaler来标准化测试集（重要！保持数据分布一致）
            test_dataset = SpectralDataset(
                test_spectral,
                test_quality,
                scaler=train_dataset.scaler  # 关键：使用训练集的标准化器
            )

            # 创建数据加载器
            calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 打印数据集大小信息
            print(f"校正集大小: {len(calibration_dataset)}")
            print(f"验证集大小: {len(validation_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")

            # 检查GPU是否可用
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {device}")

            # 创建模型实例
            model = CNN_1D_Spectra(in_dim=spectral_length, fc2_size=fc2_size ).to(device)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加L2正则化

            # 训练模型（使用校正集和验证集）
            print("\n开始训练模型...")
            model, history = train_model(
                model, calibration_loader, validation_loader, criterion, optimizer, device, num_epochs=num_epochs
            )

            save_path = f'CNN_1D_Spectra_models/{xlabel}_cnn_1Dspectra_fc2_{fc2_size}_model_{ylabel_idx}.pth'
            # 保存模型
            torch.save(model.state_dict(), save_path)
            print(f"\n模型已保存为 {save_path}")

def test(xlabel = 'nir'):

    batch_size = 32

    from load_datas import load_nir_ys

    cal_idx, test_idx, nir_Xs, ys = load_nir_ys()

    #from sklearn.preprocessing import MinMaxScaler
    #scaler_ys = MinMaxScaler()


    #cal_idx = (cal_idx-1)[0]
    #test_idx = (test_idx-1)[0]
    import numpy as np


    for ylabel_idx in [4]:#[0,1,2,3]:
        for fc2_size in [32,48,64,128,256]:

            train_spectral = nir_Xs[cal_idx,:]
            test_spectral  = nir_Xs[test_idx,:]


            ys_scaler = MinMaxScaler()
            ys_scale =  ys_scaler.fit_transform( (ys[:, ylabel_idx]).reshape(-1,1) )

            train_quality = ys_scale[cal_idx,:] #ys[cal_idx,:]
            test_quality =  ys_scale[test_idx,:] #ys[test_idx,:]

            #train_quality = train_quality[:, ylabel_idx].reshape(-1,1)
            #test_quality = test_quality[:, ylabel_idx].reshape(-1,1)

            spectral_length = train_spectral.shape[1]  # 光谱长度

            train_dataset = SpectralDataset(train_spectral, train_quality)

            # 将训练集按4:1划分为校正集和验证集
            calibration_size = int(0.6 * len(train_dataset))  # 4/5作为校正集
            validation_size = len(train_dataset) - calibration_size  # 1/5作为验证集

            calibration_dataset, validation_dataset = random_split(
                train_dataset, [calibration_size, validation_size]
            )

            # 使用训练集的scaler来标准化测试集（重要！保持数据分布一致）
            test_dataset = SpectralDataset(
                test_spectral,
                test_quality,
                scaler=train_dataset.scaler  # 关键：使用训练集的标准化器
            )

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 打印数据集大小信息
            #print(f"校正集大小: {len(calibration_dataset)}")
            #print(f"验证集大小: {len(validation_dataset)}")
            #print(f"测试集大小: {len(test_dataset)}")

            # 检查GPU是否可用
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #print(f"使用设备: {device}")

            criterion = nn.MSELoss()  # 回归任务使用均方误差损失

            # 创建模型实例
            model = CNN_1D_Spectra(in_dim =spectral_length, fc2_size=fc2_size).to(device)

            save_path = f'CNN_1D_Spectra_models/{xlabel}_cnn_1Dspectra_fc2_{fc2_size}_model_{ylabel_idx}.pth'

            model.load_state_dict(torch.load(save_path ,  weights_only=True))
            train_res = test_model(model, train_loader, criterion, device, scaler_y= ys_scaler, isShowYs=False)
            test_res = test_model(model, test_loader, criterion, device, scaler_y= ys_scaler, isShowYs=False)

            print(f'{save_path}\n 训练集:{train_res}\n  测试集:{test_res}')

def test_best_models():
    xlabel = 'nir'
    batch_size = 32

    from load_datas import load_nir_ys

    cal_idx, test_idx, nir_Xs, ys = load_nir_ys()

    #from sklearn.preprocessing import MinMaxScaler
    #scaler_ys = MinMaxScaler()


    #cal_idx = (cal_idx-1)[0]
    #test_idx = (test_idx-1)[0]
    import numpy as np
    fc2_sizes = [64, 64, 48, 256, 256]

    for ylabel_idx in [0,1,2,3,4]:
        #for fc2_size in [32,48,64,128,256]:
        fc2_size = fc2_sizes[ylabel_idx]

        train_spectral = nir_Xs[cal_idx,:]
        test_spectral  = nir_Xs[test_idx,:]

        train_quality = ys[cal_idx,:]
        test_quality = ys[test_idx,:]
        train_quality = train_quality[:, ylabel_idx].reshape(-1,1)
        test_quality = test_quality[:, ylabel_idx].reshape(-1,1)

        ys_scaler = None
        if ylabel_idx == 4:
            ys_scaler = MinMaxScaler()
            ys_scale = ys_scaler.fit_transform((ys[:, ylabel_idx]).reshape(-1, 1))
            train_quality = ys_scale[cal_idx, :]
            test_quality = ys_scale[test_idx, :]


        spectral_length = train_spectral.shape[1]  # 光谱长度

        train_dataset = SpectralDataset(train_spectral, train_quality)
        '''
        # 将训练集按4:1划分为校正集和验证集
        calibration_size = int(0.6 * len(train_dataset))  # 4/5作为校正集
        validation_size = len(train_dataset) - calibration_size  # 1/5作为验证集

        calibration_dataset, validation_dataset = random_split(
            train_dataset, [calibration_size, validation_size]
        )
        '''
        # 使用训练集的scaler来标准化测试集（重要！保持数据分布一致）
        test_dataset = SpectralDataset(
            test_spectral,
            test_quality,
            scaler=train_dataset.scaler  # 关键：使用训练集的标准化器
        )

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
        #validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 打印数据集大小信息
        #print(f"校正集大小: {len(calibration_dataset)}")
        #print(f"验证集大小: {len(validation_dataset)}")
        #print(f"测试集大小: {len(test_dataset)}")

        # 检查GPU是否可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(f"使用设备: {device}")

        criterion = nn.MSELoss()  # 回归任务使用均方误差损失

        # 创建模型实例
        model = CNN_1D_Spectra(in_dim =spectral_length, fc2_size=fc2_size).to(device)

        save_path = f'CNN_1D_Spectra_models/{xlabel}_cnn_1Dspectra_fc2_{fc2_size}_model_{ylabel_idx}.pth'

        model.load_state_dict(torch.load(save_path ,  weights_only=True))
        train_res = test_model(model, train_loader, criterion, device)
        test_res = test_model(model, test_loader, criterion, device, scaler_y=None)
        if ylabel_idx == 4:
            test_res = test_model(model, test_loader, criterion, device, scaler_y = ys_scaler)
        print(f'{save_path}\n 训练集:{train_res}\n  测试集:{test_res}')


if __name__ == "__main__":
    import torch.nn.functional as F  # 仅在主程序中导入，避免循环导入问题
    code = 0
    if code == 0:
        xlabel = 'nir'
        #main(xlabel)
        #test(xlabel)

        test_best_models()

    elif code == 1:
        # 测试最佳结果，保存预测值

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        ylabels = ['kxl', 'yd', 'kzqd', 'c1', 'c2']  # 'zmd',
        fc2_sizes = [64,64,48,256,256]     # [64,64,48,256,256]-备份
        from load_datas import load_nir_ys
        import scipy.io

        cal_idx, test_idx, nir_Xs, ys = load_nir_ys()
        spectral_length = nir_Xs.shape[1]
        #train_dataset = SpectralDataset(nir_Xs, ys)

        for idx, fc2_size in enumerate(fc2_sizes):
            # ./备份
            save_path = f'./CNN_1D_Spectra_models/nir_cnn_1Dspectra_fc2_{fc2_size}_model_{idx}.pth'
            # 创建模型实例
            model = CNN_1D_Spectra(in_dim=spectral_length, fc2_size=fc2_size).to(device)
            model.load_state_dict(torch.load(save_path, weights_only=True))

            train_dataset = SpectralDataset(nir_Xs, (ys[:, idx]).reshape(-1,1)  )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            # 检查GPU是否可用
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            criterion = nn.MSELoss()  # 回归任务使用均方误差损失

            if idx < 4:
                train_res = test_model(model, train_loader, criterion, device, isShowYs=True)

            if idx == 4:
                ys_scaler = MinMaxScaler()
                ys_scale = ys_scaler.fit_transform((ys[:, idx]).reshape(-1, 1))
                train_res = test_model(model, train_loader, criterion, device, isShowYs=True, scaler_y= ys_scaler)


            #ytrue = train_res['true_values']
            ypred = train_res['predictions']

            numpy_ypred = ypred #ypred.detach().cpu().numpy()
            print(numpy_ypred.shape)
            scipy.io.savemat(f'ypred_{idx}.mat', {'ypred': numpy_ypred})
            '''
            ypred_cal = ypred[0, cal_idx]#.detach().cpu().numpy()
            ypred_test = ypred[0, test_idx]#.detach().cpu().numpy()
            ycal = ys[cal_idx, idx]
            ytest = ys[test_idx, idx]
            
            r2c = r2_score(ycal, ypred_cal)
            r2p = r2_score(ytest, ypred_test)
            rmsec = np.sqrt( mse(ycal, ypred_cal))
            rmsep = np.sqrt( mse(ytest, ypred_test))

            res = {'r2c': r2c, 'rmsec': rmsec, 'r2p':r2p, 'rmsep': rmsep}
            print(f'指标{idx}\n{res}')
    '''


    elif code == 2:
        # 获取特征fs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        #ylabels = ['kxl', 'yd', 'kzqd', 'c1']  # , 'c2']  # 'zmd',
        fc2_sizes = [64, 64, 48, 256, 256]
        from load_datas import load_nir_ys
        import scipy.io

        cal_idx, test_idx, nir_Xs, ys = load_nir_ys()
        spectral_length = nir_Xs.shape[1]
        nir_Xs = torch.from_numpy(nir_Xs).float()
        nir_Xs = nir_Xs.unsqueeze(1)
        nir_Xs = nir_Xs.to(device)
        #train_dataset = SpectralDataset(nir_Xs, ys)
        global_pool = nn.AdaptiveAvgPool1d(1)
        for idx, fc2_size in enumerate(fc2_sizes):

            save_path = f'CNN_1D_Spectra_models/nir_cnn_1Dspectra_fc2_{fc2_size}_model_{idx}.pth'
            # 创建模型实例
            model = CNN_1D_SpectraGetFs(in_dim=spectral_length, fc2_size=fc2_size).to(device)
            model.load_state_dict(torch.load(save_path, weights_only=True))
            model.eval()
            nir_fs = model.getFeatures(nir_Xs)
            nir_fs = global_pool(nir_fs).view(nir_fs.size(0), -1)
            nir_fs = nir_fs.detach().cpu().numpy()
            scipy.io.savemat(f'nirFS_{idx}.mat', {'nir_fs': nir_fs})