function rtpd=RelativePowerDensityRatio(ton, rms, winSize, statStepSec, stepSec)
% function rtpd=RelativePowerDensityRatio(ton, rms, winSize, step, statStep)
% calculates Relative Power Density Ratio as described in 
% NOISE ROBUST FEATURES FOR SPEECH/MUSIC DISCRIMINATION IN REAL-TIME TELECOMMUNICATION
%
% ton and rms are the tonality and rms energy respectively
% winSize and step are the window size and step used to calculate rtpd

%tonalityThreshold=40;
tonalityThreshold=1;

rms1=rms;
rms1(ton<tonalityThreshold)=NaN;

rtpd=zeros(ceil(length(rms1)/(statStepSec*11025/256)+1),1);

j=1;
idx=1;
while idx<=length(rms1)
    rng=max(1, idx-floor(winSize/2)):min(idx+floor(winSize/2),length(rms1));
    t1=rms1(rng,:);
    t=rms(rng,:);
    rtpd(j)=(nanmean(t1,1))./(mean(t,1)+eps);
    idx=round(j*statStepSec/stepSec+1);
    j=j+1;
end    


rtpd(j:end,:)=[];

rtpd(isnan(rtpd))=0;

