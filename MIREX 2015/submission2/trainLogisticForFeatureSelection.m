function error=trainLogisticForFeatureSelection(trainFeat, trainOut, testFeat, testOut)
% function accuracy=trainLogisticForFeatureSelection(trainFeat, trainOut, testFeat, testOut)
%  trains a logistic model on the training set and evaluates on the test
%  set, returning the overall accuacy. Used with sequentialfs for greedy
%  feature selection
% outputs should be in the range 1...N.


model=mnrfitForSelection(trainFeat,trainOut,'model','nominal','interactions','on');	
modelvals=mnrval(model,testFeat);

[~,classifications]=max(modelvals,[],2);

error=nnz(testOut~=classifications);

%[~, accuracy]=evaluateMulticlassClassifier(classifications, testOut-1, 0);
%accuracy=accuracy(3); % F1 measure
return;
