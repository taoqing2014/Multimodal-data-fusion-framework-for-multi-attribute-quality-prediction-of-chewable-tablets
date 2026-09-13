function [RMSE,R2,bias] = rmse(y_ref,y_pred,nlv)
% RMSE 模型评价指标计算（适配红外PLS定量，区分RMSEC/RMSEP）
% 输入：
%   y_ref  : 真实参考值 一维向量
%   y_pred : 模型预测值 一维向量
%   nlv    : (可选)PLS潜变量数，输入则计算RMSEC，不输入计算RMSEP
% 输出：
%   RMSE   : 均方根误差（无nlv=RMSEP，有nlv=RMSEC）
%   R2     : 标准回归决定系数（拟合优度，论文标准）
%   bias   : 平均偏差
% 作者：修正原版错误，符合ASTM E 1655+光谱期刊通用标准

% 1. 输入校验
if numel(y_ref) ~= numel(y_pred)
    error('输入向量y_ref与y_pred样本数量不一致');
end
y_ref = y_ref(:);
y_pred = y_pred(:);
n = length(y_ref);
sse = sum((y_ref - y_pred).^2); % 残差平方和

% 2. 计算标准拟合优度R?（替代原corrcoef平方错误写法）
y_mean = mean(y_ref);
sst = sum((y_ref - y_mean).^2); % 总平方和
if sst < 1e-12
    R2 = 0;
else
    R2 = 1 - sse / sst;
end

% 3. 计算bias
bias = mean(y_ref - y_pred);

% 4. 计算RMSEC / RMSEP
if nargin == 2
    % 预测集 RMSEP
    RMSE = sqrt(sse / n);
else
    % 校正集 RMSEC，增加分母合法性保护
    df = n - nlv - 1;
    if df <= 0
        error('样本数过少或潜变量设置过大，自由度df = n-nlv-1 ≤ 0');
    end
    RMSE = sqrt(sse / df);
end
end