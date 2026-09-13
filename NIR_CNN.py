import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import ylabel
#from setuptools.sandbox import save_path
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

        self.fc_1 = nn.Linear(32 * self.conv_out_dim, fc2_size)


    def forward(self, x):

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        #self.conv_out_dim = out.size()[2]
        out = out.view(out.size(0), -1)         # 拉直 维度:16 * self.conv_out_dim

        out = self.FC_12(out)
        return out

    def getFeatures(self,x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        # self.conv_out_dim = out.size()[2]
        out = out.view(out.size(0), -1)  # 拉直 维度:16 * self.conv_out_dim
        out = self.fc_1(out)
        return out


class NIR_CNN_FeatureExtractor(nn.Module):

    def __init__(self, target_dim=32):
        super(NIR_CNN_FeatureExtractor, self).__init__()

        self.conv_1 = CNN_ConvUnit1D(1, 8,   kernel_size=3)  # 1*N*1 ->  4*(N-1)*1                               N=  256->255
        self.conv_2 = CNN_ConvUnit1D(8, 16,  kernel_size=3)  # 4*(N-1)/2*1 ->  8*(N-1)/2-1*1                        127->126
        self.conv_3 = CNN_ConvUnit1D(16, target_dim, kernel_size=3)  # 63->62

        self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.AdaptiveAvgPool1d(out)
        # self.conv_out_dim = out.size()[2]
        out = out.view(out.size(0), -1)  # 拉直 维度: B*32

        #out = self.FC_12(out)
        return out

'''
# 1. 构造模拟输入（符合 NIR 数据的单通道 1D 结构）
batch_size = 4  # 批量大小
spectrum_length = 256  # 光谱长度（对应原代码注释的 N=256）
x = torch.randn(batch_size, 1, spectrum_length)  # 随机生成模拟数据，形状 (4,1,256)

fs = NIR_CNN_FeatureExtractor()

res = fs(x)
'''
