import scipy.io

def load_nir_ys():
    # 读取.mat文件
    mat_data = scipy.io.loadmat('nir_snv_data.mat')  # 替换为你的.mat文件路径
    cal_idx = mat_data['cal_idx']
    test_idx = mat_data['test_idx']

    nir_Xs = mat_data['snv_nir_Xs']
    ys = mat_data['ys']

    cal_idx = (cal_idx )[0]
    test_idx = (test_idx)[0]

    return cal_idx, test_idx, nir_Xs, ys


