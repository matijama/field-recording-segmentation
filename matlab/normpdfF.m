function y=normpdfF(x,MU,SIGMA,noscale)
%y = normpdfF(X,MU,SIGMA,noscale)

if (nargin<4)
    noscale=0;
end

if (length(MU)==1)
    xn = (x - MU) ./ SIGMA;
else
    xn=bsxfun(@rdivide,bsxfun(@minus,x,MU),SIGMA);
end

if noscale==0
    y = bsxfun(@rdivide,exp(-0.5 * xn .^2),(sqrt(2*pi) *SIGMA));
else
    y = exp(-0.5 * xn .^2);
end


