function snv_spectra = snv(spectra)
% SNV 执行标准正态变量变换
%   spectra: 输入光谱矩阵，每行代表一个样本的光谱
%   snv_spectra: 变换后的光谱矩阵

    [n_samples, n_wavelengths] = size(spectra);
    
    % 初始化变换后的光谱矩阵
    snv_spectra = zeros(size(spectra));
    
    % 对每个样本执行SNV变换
    for i = 1:n_samples
        % 计算均值和标准差
        mean_val = mean(spectra(i,:));
        std_val = std(spectra(i,:));
        
        % 应用变换: (光谱 - 均值) / 标准差
        snv_spectra(i,:) = (spectra(i,:) - mean_val) / std_val;
    end
end