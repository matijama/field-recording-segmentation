function [selectedFeatures, bestFitModel, accuracy, outMap, outputs]=forwardFeatureSelectionLogistic(features, output, numFeaturesToSelect, useBalancedData, maxInstancesPerClass)

    featureCorr=corrcoef(features);
    
    if (useBalancedData~=0)
        [features, output] = makeBalancedFeatureSet(features, output, maxInstancesPerClass);
    end
    
    % map outputs to 1...n, needed for Logistic training
    outputs=unique(output);
    outMap=zeros(size(outputs));
    for i=1:length(outputs)
        outMap(output==outputs(i)) = i;
    end

    % partition into 1/3 test, 2/3 train
    %part = cvpartition(output,'holdout',1/3);
    part = cvpartition(output,'kfold',3);
    opt=statset;
    %opt=statset('Display','iter','UseParallel','always');
    selectedFeatures=[];
    removedFeatures=[];
    accuracy=zeros(1,numFeaturesToSelect);
    
    for i=1:numFeaturesToSelect
        model = sequentialfs( ...
            @(x,y,z,w) trainLogisticForFeatureSelection(x,y,z,w), ...
            features, ...
            outMap, ...
            'cv', part, ...
            'keepin', selectedFeatures, ...
            'keepout', removedFeatures, ...
            'nfeatures', length(selectedFeatures)+1, ...
            'options',opt ...
            );
        
        selectedFeatures = [selectedFeatures setdiff(find(model),selectedFeatures)];
        % remove all with correlation coefficient > 0.8
        removedFeatures=setdiff(find(max(abs(featureCorr(selectedFeatures, :)),[],1)>0.8),selectedFeatures);
                        
        % newly repartition data
        part=repartition(part);
		
      bestFitModel = mnrfit(features(:,selectedFeatures),outMap,'model','nominal','interactions','on');
      modelvals=mnrval(bestFitModel,features(:,selectedFeatures));
      [~,acc]=evaluateMulticlassClassifier(modelvals, outMap-1, 0);

      accuracy(i)=acc(3); % f1 measure
      %fprintf('%d ',selectedFeatures);
      %fprintf('\n');
      %fprintf('%4.2f ',accuracy(1:i));
      %fprintf('\n');
    end    
    
    
