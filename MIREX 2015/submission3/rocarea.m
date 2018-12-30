function roc=rocarea(classVals, trueVals)
% function roc=rocarea(classVals, trueVals)
% calculates the roc curve area 
% classVals are values classified by a classifier (0-1 range). Columns
%   represent probabilities of different classes
% trueVals are true class values; labels should be from 0 onwards

roc=zeros(1,size(classVals,2));
for c=1:length(roc)
    x=zeros(101,1);
    y=zeros(101,1);
    i=1;
    for tr=0:0.01:1 % threshold
        tp=nnz(classVals(:,c)>=tr & trueVals==(c-1));
        fp=nnz(classVals(:,c)>=tr & trueVals~=(c-1));
        tn=nnz(classVals(:,c)<tr & trueVals~=(c-1));
        fn=nnz(classVals(:,c)<tr & trueVals==(c-1));

        recall=tp/(tp+fn);
        fpr=fp/(fp+tn);
        x(i)=fpr;
        y(i)=recall;
        i=i+1;
    end
    [a,b]=sort(x);
    x=x(b); y=y(b);
    roc(c)=sum(diff(x).*y(1:end-1))+sum(diff(x).*(y(2:end)-y(1:end-1))/2);    
end