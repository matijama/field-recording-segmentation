function [stats, ftNames, time, energyPerc, energyPerc2, et, featureSr]=calcFeatures(wav,sr,winSize,useMedian,useEnergy,statWinSizeSec, statStepSec)
% function [stats, ftNames, time, energyPerc, energyPerc2, et, featureSr]=calcFeatures(wav,sr,winSize,useMedian,useEnergy,statWinSizeSec, statStepSec)
%
% Calculates features for classification of segments. 
% wav is the wav file
% sr is the sample rate
% winSize is the window size for feature calculation in seconds. Step will be winSize/2. If not given, it's 0.05s .
% Statistics step is ~0.5 second
% if useMedian is true, median and mad are used instead of mean and std
% if useEnergy is given: 
%  0 = features with low energy frames are considered 0, 
%  1 = features with low energy frames are taken fully
%  NaN = features with low energy frames are ignored
% statWinSizeSec is the length of statistics window in seconds
% returns array of feature stats and an array of feature names:
%  1 = std(energy)^2/mean(energy)^2
%  2 = max(zcr)/mean(zcr)
%  3 = std(spectral entropy)^2
%  4 = mean(mfcc(1))
%  5:n = std(mfcc(2:n))^2

if (sr~=11025)
%   wav=resample(wav,11025,sr);
%   sr=11025;
end
%wav=s1(wav);

if (nargin<3)
    winSize=0.05;
end
if (nargin<4)
  useMedian=0;
end
if nargin<5
    useEnergy=NaN;
end
if nargin<6
  statWinSizeSec=3.0418; %131 with 1024/44100 stepsize
end
if nargin<7
  statStepSec=20*256/11025; % 20 frames by default, cca 0.46 sec
end

winSize=pow2(floor(log2(winSize*sr)));
step=winSize/2;
featureSr=sr/step;

%% calculate features

%cepstrum
[ceps,vals]=melcepst(wav,sr,'M',10,27,winSize,step,0,5000/sr,'HpT4OS');
H=vals{1};
pd=vals{2};
%ton=max(0,vals{3}-20);
ton=vals{3};
fourHz=vals{4};
ton1=vals{5};
clear vals;

%energy and zcr
energy=zeros(size(ceps,1),1);

j=1;
for i=1:step:length(wav)-winSize+1
    if (isinteger(wav))
        t=double(wav(i:i+winSize-1))/32768;
    else
        t=wav(i:i+winSize-1);
    end   
    energy(j)=sum(t.^2)/winSize;
    j=j+1;
end

noiseFloor=zeros(size(energy));
nfs=[];
j=1;
for i=1:featureSr*10:length(energy)
  rng=max(1,floor(i-10*featureSr)):min(length(energy),ceil(i+10*featureSr));
  t=energy(rng); 
  tt=ton(rng);
  ttr=prctile(t,20);
  if any(t<ttr & tt<1)
    nf=prctile(t(t<ttr & tt<1),20);
  else
    nf=10^(-5.5);
  end
  nfs(j)=nf;
  j=j+1;
end

nfs=max(nfs,10^(-5.5));
ngt=prctile(nfs(nfs>10^(-5.5)),5);
nfs(:)=ngt;

%nfs=[zeros(1,30)+ngt nfs zeros(1,30)+ngt];
%nfs=percfilt1(nfs,5,60);
%nfs=nfs(31:end-30);

j=1;
for i=1:featureSr*10:length(energy)
  noiseFloor(floor(i):min(length(energy),ceil(i+10*featureSr)))=nfs(j);
  j=j+1;
end

features=[energy H pd ton ton1 fourHz ceps];
names={'Eng','H','PD','Ton','Ton1','fourHz'}; for i=1:size(ceps,2), names{i+6}=['Mfcc' num2str(i)]; end

features=[features deltaFeatures(features,5)];
t=length(names);
for i=1:t, names{i+t}=['d' names{i}]; end
% 
%   stats=features;
%   ftNames=names;
%   return;
%% calculate feature stats

statWinSize=round(statWinSizeSec/(step/sr));
%statStep=0.5; % 1 second
%statWinSize=131*1;
%statStep=round(statStepSec*sr/step)*step/sr;

% estimate noisefloor


[B,A]=butter(2,0.02);
%[B,A]=butter(2,0.015);
%energyF=filtfilt(B,A,sqrt(energy)).^2;
energyF=medfilt1(energy,round(2*featureSr));
%energyF=filter(B,A,sqrt(energy)).^2;
%et=energy<10^-5 | energy<energyF/10 | energy<noiseFloor*10^(3/10) ; % -55 dB or -10dB below average or below noiseFloor+3dB
et=energy<10^-5 | energy<energyF/10 | energy<noiseFloor*10^(8/10) ; % -55 dB or -10dB below average or below noiseFloor+5dB
%et=energy<10^-5.5;

rpd=RelativePowerDensityRatio(ton,sqrt(energy),statWinSize,statStepSec, step/sr);
[vsf,vsfNames]=VoicedSoundFeatures(ton,et,statWinSize,statStepSec, step/sr);

stats=zeros(ceil(step/sr*length(energy)/statStepSec)+1,size(features,2)*5);
energyPerc=zeros(size(stats,1),1);
energyPerc2=zeros(size(stats,1),1);
energyTrans=zeros(size(stats,1),1);

ftNames=cell(1,size(features,2)*5);
t=length(names);
for i=1:t, ftNames{i}=[names{i} '_mean']; end
for i=1:t, ftNames{i+t}=[names{i} '_var']; end
for i=1:t, ftNames{i+2*t}=[names{i} '_var_m2']; end
for i=1:t, ftNames{i+3*t}=[names{i} '_std']; end
for i=1:t, ftNames{i+4*t}=[names{i} '_mabs']; end
%for i=1:t, ftNames{i+5*t}=[names{i} '_kur']; end
%for i=1:t, ftNames{i+6*t}=[names{i} '_skew']; end
%for i=1:t, ftNames{i+3*t}=['mx/m_' names{i}]; end

j=1;
idx=1;
while idx<=length(energy)
    rng=max(1, idx-floor(statWinSize/2)):min(idx+floor(statWinSize/2),length(energy));
    t=features(rng,:);
    if (isnan(useEnergy))
        t(et(rng),:)=[];
    else
        t(et(rng),:)=t(et(rng),:)*useEnergy;
    end
    if ~isempty(t) && size(t,2)>2
        if useMedian==0
            tt=mean(t,1);    
            tt1=std(t,0,1).^2;
            tt2=mean(t.^2,1);
            tt5=mean(abs(t),1);
            %tt6=kurtosis(t,1,1);
            %tt7=skewness(t,1,1);
        else
            tt=median(t,1);    
            tt1=mad(t,1,1).^2;
            tt2=median(t.^2,1);
            tt5=median(abs(t),1);
            %tt6=kurtosis(t,1,1);
            %tt7=skewness(t,1,1);
        end
        tt2(tt2==0)=eps;
        %tt3=tt1./tt.^2;
        tt3=tt1./tt2;
        tt3(tt3>100)=100;
    %    tt3=max(abs(t))./abs(tt); 
    %    tt3(tt3>100)=100;
        stats(j,:)=[tt tt1 tt3 sqrt(tt1) tt5 ];
    end    
    energyPerc(j)=1-nnz(et(rng))/length(rng);
    rng2=max(1, idx-floor(statWinSize/4)):min(idx+floor(statWinSize/4),length(energy));
    energyPerc2(j)=1-nnz(et(rng2))/length(rng2);
    energyTrans(j)=nnz(diff(et(rng)))/length(rng);
    idx=round(j*statStepSec*sr/step+1);
    j=j+1;
end    
stats(j:end,:)=[];
energyPerc(j:end)=[];
energyPerc2(j:end)=[];
energyTrans(j:end)=[];
stats=[stats rpd vsf energyPerc energyTrans];
ftNames=[ftNames 'rpd' vsfNames 'energyPerc', 'energyTrans'];
time=(0:size(stats,1)-1)*statStepSec;
%     t=features(i:i+statWinSize-1,1);
%     tt=mean(t);
%     if (tt~=0)
%         stats(j,1)=std(t)^2/tt^2;
%     end;
% 
%     t=features(i:i+statWinSize-1,2);
%     tt=mean(t);
%     if (tt~=0)
%         stats(j,2)=max(t)/tt;
%     end
%     
%     stats(j,3)=std(features(i:i+statWinSize-1,3))^2;
%     
%     t=features(i:i+statWinSize-1,4:end);
%     stats(j,4)=mean(t(:,1));
%     stats(j,5:size(stats,2))=std(t(:,2:size(t,2))).^2;
%     j=j+1;
%end
