function [PLS_result] = tq_PLSR_cal_test(X, y, cal_idx, test_idx)

    A=10;
    K=5; 
    method='center';
    
    Xcal=X(cal_idx,:);
    ycal=y(cal_idx,:);
    Xtest=X(test_idx,:);
    ytest=y(test_idx,:);
    
    
    CV=plscv(Xcal,ycal,A,K,method);
    nvl = CV.optLV;
    fm.RMSECV = CV.RMSECV_min;

    PLS=pls(Xcal,ycal,nvl);  %+++ Build a PLS regression model using training set
    [ypred,RMSEP]=plsval(PLS,Xtest,ytest); %+++ make predictions on test set
    [fm.RMSEC,fm.R2c,fm.biasc]=rmse(ycal,PLS.y_fit);
    %[fm.RMSEP,fm.R2p,fm.biasp]=rmse(ytest,ypred);
    [fm.RMSEP,fm.R2p,fm.biasp] = rmse(ytest, ypred);
    fm.nLV = nvl;
    fm.PLS = PLS;
    fm.RPD = std(ytest)/RMSEP;
    
    fm.ycal = ycal;
    fm.ytest = ytest;
    fm.ypred_test = ypred;
    [fm.ypred_cal,RMSEC]=plsval(PLS,Xcal,ycal);
    PLS_result = fm;
    
    
end