function [segTimes, segLabels, segClasses, segClassSr]=segmentRecordingDeep(filename, segClasses)
%function [segTimes, segLabels]=segmentRecording(filename, segClasses)
% 
% segments a recording and returns segment times and labels
% segment times are in seconds
% filename is the audio file to segment
% segClasses are the probabilities of classification of the file contents with 1/samplerate of statStepInSec
%      e.g.  % calculated with a deep classification model

energyWinInSec=1024/22050;
energyMedianInSec=1;
statStepInSec=0.5;
statWinInSec=2;
klWinSizeInSec=5;
klGaussWinInSec=5; 
klPeakThresh=4;

%% read file
[w, sr] = audioread(filename);
w=s1(mean(w,2));
if (sr~=22050)
  w=resample(w,22050,sr);
  sr=22050;
end

%% calculate energy 
[energy, et, featureSr]=calcEnergy2018(w,sr,energyWinInSec,statWinInSec, statStepInSec);

segClasses(end+1:length(energy),:)=repmat(segClasses(end,:),length(energy)-size(segClasses,1),1);

etM=medfilt1(et+0.0,round(energyMedianInSec*featureSr));
etStat=interp1((0:length(et)-1)/featureSr, etM, (0:length(energy)-1)'*statStepInSec,'linear','extrap');

segClassSr=1/statStepInSec;

%% calculate transitions with KL on segClasses
klC=zeros(size(segClasses,1),1);
klWinSize=round(klWinSizeInSec/statStepInSec);

for st=round(klWinSize/2)+1:size(segClasses,1)-round(klWinSize/2)
  rng=max(st-klWinSize*2+1,1):st;
  X=segClasses(rng,:); X(etStat(rng)>0.5,:)=[];  
  Xe=sqrt(energy(rng,:)); Xe(etStat(rng)>0.5)=[];
  rng=st+1:min(size(segClasses,1),st+klWinSize*2);
  Y=segClasses(rng,:); Y(etStat(rng)>0.5,:)=[];  
  Ye=sqrt(energy(rng,:)); Ye(etStat(rng)>0.5)=[];
  X=X(max(1,size(X,1)-klWinSize):end,:);
  Xe=Xe(max(1,length(Xe)-klWinSize):end);
  Y=Y(1:min(size(Y,1),klWinSize),:);
  Ye=Ye(1:min(length(Ye),klWinSize));

  h1=max(sum(bsxfun(@times,X,Xe),1)/sum(Xe),1e-3);
  h2=max(sum(bsxfun(@times,Y,Ye),1)/sum(Ye),1e-3);
  kl=sum(h1.*log2(h1+eps)-h1.*log2(h2+eps))+sum(h2.*log2(h2+eps)-h2.*log2(h1+eps));  

  klC(st)=kl;  
end

klGaussWin=klGaussWinInSec/statStepInSec;
klGaussWin=klGaussWin+(~mod(klGaussWin,2));
t=gausswin(klGaussWin); t=t/sum(t);
klCG=conv(klC,t,'same');

%% calculate silence and transition likelihoods
[b,a]=butter(2,0.1*statStepInSec*2);

% weigh energy according to class, so that e.g. for speech only longer silence  makes new segments
etShift=[1.2 1 0.9 1.5]*segClasses';
energyCurve=filtfilt(b,a,etStat.*etShift'+(1-etShift'))';
energyCurve=min(max(energyCurve,eps),1);
klCurve=min(max(filtfilt(b,a,(klCG'>klPeakThresh)+0.0),eps),1);

%% do the probabilistic segmentation

% loThresh(2), minLength, bB, bLen
P2=[0 0 10/statStepInSec 1 1];
loc=segEKProbability_2018(energyCurve, klCurve,segClasses, P2, statStepInSec);

%% calculate segment labels
segTimes=loc;
segLabels=zeros(length(segTimes)-1,size(segClasses,2));
for i=1:length(segTimes)-1
    segLabels(i,:)=sum(bsxfun(@times,segClasses(segTimes(i):segTimes(i+1),:),(energy(segTimes(i):segTimes(i+1)).^0.5).*hamming(segTimes(i+1)-segTimes(i)+1)));
    segLabels(i,:)=segLabels(i,:)/(sum(segLabels(i,:))+eps);
end

segTimes=statStepInSec*(segTimes-1);
segTimes(2:end-1)=segTimes(2:end-1)+statWinInSec/2;

