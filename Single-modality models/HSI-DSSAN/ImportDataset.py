import numpy as np
import spectral
import torch

from pathlib import Path

# __file__：当前这个py文件
# .parent.parent：往上两级目录（对应你之前的../../）
base = Path(__file__).parent.parent.parent
BATCH_DATA_PATH = base / f"roi-1000-2500"
BATCH_DATA_PATH = str(BATCH_DATA_PATH) +  "/"
#BATCH_DATA_PATH = f'../../code+data/roi-1000-2500/'

def load_hsi_img(batch_num = 0, H_num = 1, width=20, height=20, BASE_PATH = BATCH_DATA_PATH ):
    #batch_num = 0  # 0-19
    #H_num = 1  # 1-5
    #BASE_PATH = f'../数据/高光谱数据（0-19批）/HSI-Spectra/'
    img = spectral.envi.open(f'{BASE_PATH}{batch_num}-H{H_num}.hdr', f'{BASE_PATH}{batch_num}-H{H_num}.dat')
    # 行-列-通道  ->   通道-宽（列）-高（行）
    np_img = img[0:height, 0:width, :]
    np_img = np_img.transpose(2, 1, 0)
    #np_img = np.tanh(np_img)        # ************************************ -1~1
    input_tensor = torch.from_numpy(np_img).float()
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor         # 1*C*H*W


def load_hsi_imgs(BASE_PATH = BATCH_DATA_PATH):

    hsi_imgs = []
    for batch_num in range(1, 41):      # 1-40
        for H_num in range(1, 6):       # 1-5
            itensor = load_hsi_img(batch_num = batch_num, H_num = H_num, BASE_PATH = BASE_PATH)
            if len(hsi_imgs) == 0:
                hsi_imgs = itensor
            else:
                hsi_imgs = torch.cat((hsi_imgs, itensor))
    return hsi_imgs


#load_hsi_imgs()