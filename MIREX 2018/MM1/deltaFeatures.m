function y=deltaFeatures(x,winSize)
%function y=deltaFeatures(x,winSize)
%
% calculates first order delta features by estimating the linearTrend within the specified window
% x is (sample, feature) matrix (features are columns)
% winSize should be odd. 

    lr=(winSize-1)/2;
    nr=size(x,1);
    nc=size(x,2);

    vf=(lr:-1:-lr)/sum((lr:-1:-lr).^2);
    ww=ones(lr,1);
    cx=[x(ww,:); x; x(nr*ww,:)];
    y=reshape(filter(vf,1,cx(:)),nr+2*lr,nc);
    y(1:2*lr,:)=[];
