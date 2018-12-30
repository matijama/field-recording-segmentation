function extractFeaturesMirex15(scratchPath, fileList, numCores)
  if nargin<3
    numCores=4;
  end

  params.resample=1;
  params.winSize=0.05;
  params.useMedian=0;
  params.useEnergy=NaN;
  params.statWinSizeSec=3;
  params.statStepSec=0.25;
  featSubset=[2 3 4 5 6 7 8 9 10 11 12 38 40 41 42 43 44 50 54 68 70 98 101 113 116 120 121 122 123 124 148 152 153 154 155 161 162 163 169];  
 
  f=fopen(fileList,'r');
  files=textscan(f,'%s','delimiter','\n');
  fclose(f);
  files=files{1};
  
  poolobj = gcp('nocreate'); 
  if isempty(poolobj)
    parpool(numCores);
  end
  
  parfor i=1:length(files)
    disp(['Extracting features from ' files{i}]);
    [w,sr]=audioread(files{i});
    w=mean(w,2);
    w=s1(w);
    if (params.resample~=0)
      w=resample(w,22050,sr);
      sr=22050;
    end
    [ft,ftNames,time, energyPerc]=calcFeatures(w,sr,params.winSize,params.useMedian,params.useEnergy,params.statWinSizeSec,params.statStepSec);
    
    [~,name]=fileparts(files{i});
    sav(fullfile(scratchPath,[name '.ft.mat']),ft(:,featSubset),ftNames(featSubset),time, energyPerc);
  end
end

function sav(name,ft,ftNames,time, energyPerc)
  save(name,'ft','ftNames','time','energyPerc');
end
