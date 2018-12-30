function trainMusicSpeechMirex15a(scratchPath, fileList, numCores)
  model=[];
  selF=[];
  outMap=[];
  outputs=[];
  
  save(fullfile(scratchPath,'trainedMusicSpeechModel.mat'),'selF','model','outMap','outputs');
end

