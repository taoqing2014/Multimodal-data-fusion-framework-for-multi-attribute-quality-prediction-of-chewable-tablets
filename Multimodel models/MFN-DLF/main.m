clear;
load('res-kxl.mat');		% Porosity 
[highDataFusion_kxl_result] = tq_highDataFusion_LR_getResult(kxl, ypred_nir_cnn', ypred_hsi_DSSAN, cal_idx+1, test_idx+1 );
load('res-yd.mat');		% Hardness 
[highDataFusion_yd_result] = tq_highDataFusion_LR_getResult(yd, ypred_nir_cnn', ypred_hsi_DSSAN, cal_idx+1, test_idx+1 );
%load('res-kzqd.mat');	
%[highDataFusion_kzqd_result] = tq_highDataFusion_LR_getResult(kzqd, ypred_nir_cnn', ypred_hsi_DSSAN, cal_idx+1, test_idx+1 );
load('res-c1.mat');		% Narirutin content
[highDataFusion_c1_result] = tq_highDataFusion_LR_getResult(c1, ypred_nir_cnn', ypred_hsi_DSSAN, cal_idx+1, test_idx+1 );
load('res-c2.mat');		% Hesperidin content
[highDataFusion_c2_result] = tq_highDataFusion_LR_getResult(c2, ypred_nir_cnn', ypred_hsi_DSSAN, cal_idx+1, test_idx+1 );

res = [
    highDataFusion_kxl_result.R2c, highDataFusion_kxl_result.RMSEC, highDataFusion_kxl_result.R2p,highDataFusion_kxl_result.RMSEP, highDataFusion_kxl_result.RPD, highDataFusion_kxl_result.RMSEP/highDataFusion_kxl_result.RMSEC;
    highDataFusion_yd_result.R2c, highDataFusion_yd_result.RMSEC, highDataFusion_yd_result.R2p,highDataFusion_yd_result.RMSEP, highDataFusion_yd_result.RPD, highDataFusion_yd_result.RMSEP/highDataFusion_yd_result.RMSEC;
    %highDataFusion_kzqd_result.R2c, highDataFusion_kzqd_result.RMSEC, highDataFusion_kzqd_result.R2p,highDataFusion_kzqd_result.RMSEP, highDataFusion_kzqd_result.RPD, highDataFusion_kzqd_result.RMSEP/highDataFusion_kzqd_result.RMSEC;
    highDataFusion_c1_result.R2c, highDataFusion_c1_result.RMSEC, highDataFusion_c1_result.R2p,highDataFusion_c1_result.RMSEP, highDataFusion_c1_result.RPD, highDataFusion_c1_result.RMSEP/highDataFusion_c1_result.RMSEC;
    highDataFusion_c2_result.R2c, highDataFusion_c2_result.RMSEC, highDataFusion_c2_result.R2p,highDataFusion_c2_result.RMSEP, highDataFusion_c2_result.RPD, highDataFusion_c2_result.RMSEP/highDataFusion_c2_result.RMSEC;
  ];

%ypred_high_fusion = [highDataFusion_kxl_result.ypreds, highDataFusion_yd_result.ypreds, highDataFusion_kzqd_result.ypreds, highDataFusion_c1_result.ypreds, highDataFusion_c2_result.ypreds];
ypred_high_fusion = [highDataFusion_kxl_result.ypreds, highDataFusion_yd_result.ypreds,  highDataFusion_c1_result.ypreds, highDataFusion_c2_result.ypreds];


