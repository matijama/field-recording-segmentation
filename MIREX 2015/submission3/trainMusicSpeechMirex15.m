function trainMusicSpeechMirex15(scratchPath, fileList, numCores)
  if nargin<3
    numCores=4;
  end

  f=fopen(fileList,'r');
  t=textscan(f,'%s\t%s','delimiter','\t');
  fclose(f);
  files=t{1};
  labels=t{2};
  
  poolobj = gcp('nocreate'); 
  if isempty(poolobj)
    parpool(numCores);
  end
  
  i=1;
  [~,name]=fileparts(files{i});
  x=load(fullfile(scratchPath,[name '.ft.mat']));
  allFeatures=zeros(size(x.ft,1)*length(files),size(x.ft,2));
  allLabels=zeros(size(x.ft,1)*length(files),1);
  j=1;
  for i=1:length(files)
    [~,name]=fileparts(files{i});
    x=load(fullfile(scratchPath,[name '.ft.mat']));
    x.ft(x.energyPerc<0.2,:)=[];
    if (~isempty(x.ft))
      allFeatures(j:j+size(x.ft,1)-1,:)=x.ft;
      allLabels(j:j+size(x.ft,1)-1)=labels{i};
      j=j+size(x.ft,1);
    end
  end
  allFeatures(j:end,:)=[];
  allLabels(j:end)=[];
  disp('Training model ...');
  [selF, model, accuracy, outMap, outputs]=forwardFeatureSelectionLogistic(allFeatures, allLabels, 12, 0);
  save(fullfile(scratchPath,'trainedMusicSpeechModel.mat'),'selF','model','outMap','outputs');
end

