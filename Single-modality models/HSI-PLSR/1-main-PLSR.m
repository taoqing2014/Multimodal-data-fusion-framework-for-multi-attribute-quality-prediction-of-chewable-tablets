
clear;
load('Xy.mat');

snv_Xs = snv(Xs);
snv_PLS_result_1 = tq_PLSR_cal_test(snv_Xs, ys(:,1), cal_idx, test_idx); % Porosity 
snv_PLS_result_2 = tq_PLSR_cal_test(snv_Xs, ys(:,2), cal_idx, test_idx); % Hardness
%snv_PLS_result_3 = tq_PLSR_cal_test(snv_Xs, ys(:,3), cal_idx, test_idx);% 
snv_PLS_result_4 = tq_PLSR_cal_test(snv_Xs, ys(:,4), cal_idx, test_idx);  % Narirutin content
snv_PLS_result_5 = tq_PLSR_cal_test(snv_Xs, ys(:,5), cal_idx, test_idx);  % Hesperidin content

snv_res = [
    snv_PLS_result_1.nLV, snv_PLS_result_1.R2c, snv_PLS_result_1.RMSEC, snv_PLS_result_1.RMSECV, snv_PLS_result_1.R2p, snv_PLS_result_1.RMSEP, snv_PLS_result_1.RPD;
    snv_PLS_result_2.nLV, snv_PLS_result_2.R2c, snv_PLS_result_2.RMSEC, snv_PLS_result_2.RMSECV, snv_PLS_result_2.R2p, snv_PLS_result_2.RMSEP, snv_PLS_result_2.RPD;
    %snv_PLS_result_3.nLV, snv_PLS_result_3.R2c, snv_PLS_result_3.RMSEC, snv_PLS_result_3.RMSECV, snv_PLS_result_3.R2p, snv_PLS_result_3.RMSEP, snv_PLS_result_3.RPD;
    snv_PLS_result_4.nLV, snv_PLS_result_4.R2c, snv_PLS_result_4.RMSEC, snv_PLS_result_4.RMSECV, snv_PLS_result_4.R2p, snv_PLS_result_4.RMSEP, snv_PLS_result_4.RPD;
    snv_PLS_result_5.nLV, snv_PLS_result_5.R2c, snv_PLS_result_5.RMSEC, snv_PLS_result_5.RMSECV, snv_PLS_result_5.R2p, snv_PLS_result_5.RMSEP, snv_PLS_result_5.RPD;
];