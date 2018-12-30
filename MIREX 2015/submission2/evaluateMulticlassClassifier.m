function [table, avg, confMat]=evaluateMulticlassClassifier(classVals, trueVals, print)
% table=evaluateMulticlassClassifier(classVals, trueVals, print)
% evaluates the classifier with standard measures (as in weka)
% classVals are values classified by a classifier (0-1 range). Columns
%   represent probabilities of different classes
% trueVals are true class values from 0 on
% returns precision, recall, F1 and roc for each class (table) and its
% weigted average (Avg), as well as the confusion Matrix (confMat)

if (nargin<3)
    print=1;
end
roc=rocarea(classVals,trueVals);

[~,b]=max(classVals,[],2);
confMat=accumarray([trueVals+1 b],ones(size(trueVals,1),1),[size(classVals,2),size(classVals,2)]);			

tp=diag(confMat);
fp=sum(confMat,1)'-tp;
fn=sum(confMat,2)-tp;

precision=tp./(tp+fp+eps);
recall=tp./(tp+fn+eps);
f1=2*precision.*recall./(precision+recall+eps);

table=[precision recall f1 roc'];
t=sum(confMat,2);
avg=t'*table/sum(t);

if print ~=0
    fprintf('%10s %10s %10s %10s %10s\n','','Precision','Recall','F-Measure','ROC Area');
    for i=1:size(table,1)
        fprintf('%10s %10.3f %10.3f %10.3f %10.3f\n','',table(i,:));
    end
    fprintf('%10s %10.3f %10.3f %10.3f %10.3f\n','W. Average',avg);
    fprintf('\n'); 
end
