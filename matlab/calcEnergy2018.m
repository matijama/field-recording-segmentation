function [energyOut, et, featureSr]=calcEnergy2018(w,sr,winSizeInSec,statWinSizeInSec, statStepInSec)
% function [energyOut, et, featureSr]=calcEnergy2016(wav,sr,winSizeInSec,statWinSizeInSec, statStepInSec)
%
% Calculates features for classification of segments. 
% wav is the wav file
% sr is the sample rate
% winSizeInSec is the window size for feature calculation in seconds. Step will be winSize/2. If not given, it's 0.05s .
% Statistics step is by default 0.5 second
% statWinSizeSec is the length of statistics window in seconds
% returns energy, and 01 vector of values under noisefloor (true), and featureSampleRate

thresh=10^(-6); % -60 dB

if (nargin<3)
    winSizeInSec=0.05;
end
if nargin<4
  statWinSizeInSec=2; 
end
if nargin<5
  statStepInSec=0.5; 
end

winSize=pow2(floor(log2(winSizeInSec*sr)));
step=winSize/2;
featureSr=sr/step;
statWinSize=round(statWinSizeInSec /(step/sr));


%% RMS energy 
energy=zeros(ceil(length(w)/step),1);
j=1;
for i=1:step:length(w)-winSize+1
    if (isinteger(w))
        t=double(w(i:i+winSize-1))/32768;
    else
        t=w(i:i+winSize-1);
    end   
    energy(j)=sum(t.^2)/winSize;
    j=j+1;
end
energy(j:end)=[];



%% global and local noise floor

% global noise floor - 1st percentile
noiseFloor=RankOrderFilter(energy,round(20*featureSr),1).^0.9;
energyF=medfilt1(energy,round(featureSr));

% local noise floor
eL=medfilt1(energyF,round(12*featureSr));
energyFLT=10^(-3)*eL.^0.5; % -20 => -40, -30 => -45, -40 => -50 ...

% reset global noise floor if its too high (e.g. long regions with strong amplitude
nfReset=noiseFloor>energyFLT & noiseFloor>1e-4 & noiseFloor<0.3*eL.^0.9;
noiseFloor(nfReset)=energyFLT(nfReset);
noiseFloor(noiseFloor<thresh)=thresh;

% when are we under noise floor
et=energyF<noiseFloor | energyF<energyFLT; 


%% average the output over window

energyOut=zeros(ceil(step/sr*length(energy)/statStepInSec)+1,1);

j=1;
idx=1;
while idx<=length(energy)
    rng=idx:min(idx+floor(statWinSize)-1,length(energy));
    t=energy(rng,:);
    energyOut(j)=mean(t,1);
    idx=round(j*statStepInSec*sr/step+1);
    j=j+1;
end    
energyOut(j:end)=[];
