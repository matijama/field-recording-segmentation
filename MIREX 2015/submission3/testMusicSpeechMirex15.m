function testMusicSpeechMirex15(scratchPath, fileList, outputFile, numCores)
  if nargin<4
    numCores=4;
  end

  poolobj = gcp('nocreate'); 
  if isempty(poolobj)
    parpool(numCores);
  end

  ld=load(fullfile(scratchPath,'trainedMusicSpeechModel.mat'));  
  tm=load('trainedModels');
 
  f=fopen(fileList,'r');
  t=textscan(f,'%s','delimiter','\n');
  fclose(f);
  files=t{1};
  for i=1:length(files)
    t=strsplit(files{i},'\t');
    files{i}=t{1};
  end
  
  
  allLabels=zeros(size(files));
  disp(['Testing...']);
  parfor i=1:length(files)

    [~,name]=fileparts(files{i});

    x=load(fullfile(scratchPath,[name '.ft.mat']));
    if (~isempty(ld.model))
      [mc,sc]=doClass(x.ft,x.energyPerc, ld.model,ld.selF,ld.outputs);
      if (abs(mc-sc)<0.2)
        [mc,sc]=doClass(x.ft,x.energyPerc, [ld.model tm.model],[ld.selF tm.selF],[ld.outputs tm.outputs]);
      end
    else
      [mc,sc]=doClass(x.ft,x.energyPerc, tm.model,tm.selF,tm.outputs);
    end
    if (mc>=sc)
      allLabels(i)='m';
    else
      allLabels(i)='s';
    end    
  end
  f=fopen(outputFile,'w');
  for i=1:length(files)
    fprintf(f,'%s\t%c\n',files{i},allLabels(i));
  end
  fclose(f);  
  
end

function [mc,sc]=doClass(ft, energyPerc, model, selF, outputs)

  if (isa(model,'double'))
    allC = mnrval(model,ft(:,selF));
  elseif iscell(model)
    allC=cell(1,length(model));
    for i=1:length(model)
      allC{i} = mnrval(model{i},ft(:,selF{i}));
    end
  end

  if iscell(model)    
    for i=1:length(model)      
      meanc=sum(bsxfun(@times,allC{i},energyPerc),1)/sum(energyPerc);
      
      [ismusic,isspeech]=getCat(outputs{i});
      mc(i)=max(meanc(ismusic));
      sc(i)=max(meanc(isspeech));
      mc(i)=mc(i)/(mc(i)+sc(i)+eps);
      sc(i)=1-mc(i);
    end
    mc=sum(mc);
    sc=sum(sc);    
  else
    meanc=sum(bsxfun(@times,allC,energyPerc),1)/sum(energyPerc);

    [ismusic,isspeech]=getCat(outputs);
    mc=max(meanc(ismusic));
    sc=max(meanc(isspeech));
  end
  mc=mc/(mc+sc+eps);
  sc=1-mc;
end


function [ismusic,isspeech]=getCat(outputs)
  ismusic=(outputs=='1' | outputs=='2' | outputs=='b' | outputs =='i' | outputs=='m');
  isspeech=(outputs=='s');
  if ~any(ismusic)
    ismusic=~isspeech;
  end
  if ~any(isspeech)
    isspeech=~ismusic;
  end
end

