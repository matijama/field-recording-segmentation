function doMusicAndSpeechDetection(inFile, outFile, pars)

  if nargin<3
    pars.energyPerc=0;
    pars.energyThreshold=0.2;
    pars.klSegWindowSec=15;
    pars.klSegPeakSize=10;
    pars.klMusicWindow=15;
    pars.klMusicDif=7500;
    pars.klSpeechWindow=15;
    pars.klSpeechDif=100;
    pars.pauseRemoveSec=1;
    pars.pauseKeepSec=3;
    pars.pauseSegSec=1.5;
  end
    
  tm=load('trainedModels');

  params.resample=1;
  params.winSize=0.05;
  params.useMedian=0;
  params.useEnergy=NaN;
  params.statWinSizeSec=3;
  params.statStepSec=0.1;
  featSubset=[2 3 4 5 6 7 8 9 10 11 12 38 40 41 42 43 44 50 54 68 70 98 101 113 116 120 121 122 123 124 148 152 153 154 155 161 162 163 169];  

  [w,sr]=audioread(inFile);
  w=mean(w,2);
  w=s1(w);
  if (params.resample~=0 && sr ~=22050)
    w=resample(w,22050,sr);
    sr=22050;
  end
  [ft,ftNames,time, energyPerc, energyPerc2, et, featureSr]=calcFeatures(w,sr,params.winSize,params.useMedian,params.useEnergy,params.statWinSizeSec,params.statStepSec);

  [mc,sc,dc] = doClass(ft(:,featSubset), params.statStepSec, tm.model, tm.selF, tm.outputs);
  
  labels=nan(size(mc));

  aa=klCurve(mc,pars.klSegWindowSec,params.statStepSec);
%  a2=klCurve(sc,pars.klSegWindowSec,params.statStepSec);
  
  %aa=(a1+a2)/2;
  [~,pl]=findpeaks(aa,'MinPeakHeight',pars.klSegPeakSize,'MinPeakDistance',2/params.statStepSec);
  pl=union(pl,length(mc));
  
  loc=[];
  if (pars.energyPerc==2)
    pl=union(pl,find(energyPerc2(1:end-1)>pars.energyThreshold & energyPerc2(2:end)<=pars.energyThreshold)+1);
    labels(energyPerc2<=pars.energyThreshold)=0;
  elseif (pars.energyPerc==1)
    pl=union(pl,find(energyPerc(1:end-1)>pars.energyThreshold & energyPerc(2:end)<=pars.energyThreshold)+1);
    labels(energyPerc<=pars.energyThreshold)=0;
  else
    et=medfilt1(double(et),round(featureSr));
    loc=min(round(([find(diff([0;et])>0)-1 find(diff([et;0])<0)-2])/featureSr/params.statStepSec)+1,length(labels));
%    loc(((loc(:,2)-loc(:,1))+1<0.8/params.statStepSec) & loc(:,1)~=1 & loc(:,2)~=length(labels),:)=[]; %remove less than 0.8 second
%    t=find((loc(:,2)-loc(:,1))+1>1.5/params.statStepSec | loc(:,1)==1 | loc(:,2)==length(labels)); 
    loc(((loc(:,2)-loc(:,1))+1<pars.pauseRemoveSec/params.statStepSec) & loc(:,1)~=1 & loc(:,2)~=length(labels) & arrayfun(@(x,y) ~any(x<=pl & y>=pl), loc(:,1),loc(:,2)),:)=[]; %remove less than 0.8 second
    t=find((loc(:,2)-loc(:,1))+1>pars.pauseSegSec/params.statStepSec | loc(:,1)==1 | loc(:,2)==length(labels) | arrayfun(@(x,y) any(x<=pl & y>=pl), loc(:,1),loc(:,2))); 
    pl=union(pl,loc(t,1)); % add longer than 1.5 seconds for labeling    
    
    for i=1:length(t)      
      labels(loc(t(i),1):loc(t(i),2))=0;
    end

    loc(t,:)=[]; % remove, only keep those between 0.8 and pars.pauseSegSec seconds
  end
  
  % find segments and label them
  t=zeros(1,2);
  for j=1:length(pl)
    st=find(isnan(labels),1,'first');
    en=pl(j)-1;
    while (en>=1 && ~isnan(labels(en)))      
      en=en-1;
    end
    if (en>=st)
      t(1)=dot(mc(st:en),energyPerc(st:en))/sum(energyPerc(st:en));
      t(2)=dot(sc(st:en),energyPerc(st:en))/sum(energyPerc(st:en));
      [~,t]=max(t);
      labels(st:en)=t/2;
    end
  end
  
  t=[find(diff([-1;labels])~=0)];
  t=[t labels(t)];
  
  % remove short music/speech segments
  for i=1:size(t,1)
    if (t(i,2)~=0)
      if i<size(t,1)
        nx=t(i+1,1);
      else
        nx=length(labels)+1;
      end
      if (nx-t(i,1))<2/params.statStepSec
        if (i>1 && t(i-1,2)~=0)
          labels(t(i,1):nx)=t(i-1,2);
        elseif (i<size(t,1) && t(i+1,2)~=0)
          labels(t(i,1):nx)=t(i+1,2);
        end        
      end
    end
  end

  % add short pauses
  for i=1:size(loc,1)
    labels(loc(i,1):loc(i,2))=0;
  end
  
  t=[find(diff([-1;labels])~=0)];
  t=[t labels(t)];

  i=2;
  kls=nan(size(labels));
  while i<size(t,1)
    if t(i,2)==0 % pause
      if t(i+1,2)==t(i-1,2) && (t(i+1,1)-t(i,1))*params.statStepSec<pars.pauseKeepSec
        prev=find(t(1:i-2,2)~=0 & t(1:i-2,2)~=t(i-1,2),1,'last');
        if isempty(prev)
          prev=i-1;
        else
          while (t(prev,2)~=t(i-1,2))
            prev=prev+1;
          end
        end
        next=find(t(i+2:end,2)~=0 & t(i+2:end,2)~=t(i-1,2),1,'first');
        
        % short pause, if instrumental see whether timbre is different
        if (t(i+1,2)<0.9)
          r1=max(t(prev,1),round(t(i,1)-pars.klMusicWindow/params.statStepSec)):t(i,1)-1;
          if (isempty(next))
            r2=t(i+1,1):min([size(ft,1) length(labels) round(t(i+1,1)+pars.klMusicWindow/params.statStepSec)]);
          else
            r2=t(i+1,1):min([size(ft,1) length(labels) t(i+1+next,1)-1 round(t(i+1,1)+pars.klMusicWindow/params.statStepSec)]);
          end
          
          try
            b1=ft(r1,7:16);
            b2=ft(r2,7:16);
            b1(energyPerc(r1)<0.2,:)=[];
            b2(energyPerc(r2)<0.2,:)=[];
          catch
            b1=[]; b2=[];
          end
          if (size(b2,1)<pars.klMusicWindow/params.statStepSec*0.75 || size(b1,1)<pars.klMusicWindow/params.statStepSec*0.75)
            t(i:i+1,:)=[];
            continue;
          end
          
          kl=klSDiv(b1,b2)*(t(i+1,1)-t(i,1))*params.statStepSec;
          kls(t(i,1))=kl;
          if (kl<pars.klMusicDif)
            t(i:i+1,:)=[];
            continue;
          end
        else % speech
          r1=max(t(prev,1),round(t(i,1)-pars.klSpeechWindow/params.statStepSec)):t(i,1)-1;
          if isempty(next)
            r2=t(i+1,1):min([size(ft,1) length(labels) round(t(i+1,1)+pars.klSpeechWindow/params.statStepSec)]);
          else
            r2=t(i+1,1):min([size(ft,1) length(labels) t(i+1+next,1)-1 round(t(i+1,1)+pars.klSpeechWindow/params.statStepSec)]);
          end
          try
            b1=ft(r1,12:16);
            b2=ft(r2,12:16);
            b1(energyPerc(r1)<0.2,:)=[];
            b2(energyPerc(r2)<0.2,:)=[];
          catch
            b1=[];
            b2=[];
          end
          if (size(b2,1)<pars.klSpeechWindow/params.statStepSec*0.75 || size(b1,1)<pars.klSpeechWindow/params.statStepSec*0.75)
            t(i:i+1,:)=[];
            continue;
          end
          
          kl=klSDiv(b1,b2)*(t(i+1,1)-t(i,1))*params.statStepSec;
          kls(t(i,1))=kl;          
          if (kl<pars.klSpeechDif)
            t(i:i+1,:)=[];
            continue;
          end
        end
      end
    end
    i=i+1;
  end
  
  t=[t(:,1) [t(2:end,1);length(labels)]-t(:,1)-0.001 t(:,2)];  
  t(t(:,3)==0,:)=[];
    
  t=sortrows(t,1);
  
  if ~isempty(outFile)
    f=fopen(outFile,'w');
  else
    f=1;
  end
  for i=1:size(t,1)
    if (t(i,3)>0.9)
      v='speech';
    elseif (t(i,3)>0.4)
      v='music';
      fprintf(f,'%6.2f\t%6.2f\t%s\n',(t(i,1)-1)*params.statStepSec,((t(i,1)-1)+t(i,2))*params.statStepSec,v);
    else
      v='0';
    end
  end
  if (f~=1)
    fclose(f);     
  end
  
%   a4=klCurve(energyPerc,5,statStepSec);
%   a5=klCurve(energyPerc2,5,statStepSec);
% 
%   mc=medfilt1(mc,2/statStepSec);
%   sc=medfilt1(sc,2/statStepSec);
%   oc=medfilt1(oc,2/statStepSec);
%   energyPerc=medfilt1(energyPerc,2/statStepSec);   
  

end

function [mc,sc,dc]=doClass(ft, statStepSec, model, selF, outputs)
  

  if (isa(model,'double'))
    allC = mnrval(model,ft(:,selF));
  elseif iscell(model)
    allC=cell(1,length(model));
    for i=1:length(model)
      allC{i} = mnrval(model{i},ft(:,selF{i}));
    end
  end
  dc=[];
  if iscell(allC)    
    mc=[]; sc=[]; 
    for i=1:length(model)      
      [ismusic,isspeech]=getCat(outputs{i});
      mc1=max(allC{i}(:,ismusic),[],2);
      sc1=max(allC{i}(:,isspeech),[],2);
      mc1=mc1./(mc1+sc1+eps);
      sc1=1-mc1;
      %dc1=diffCurve(zscore(ft(:,selF{i}),0,1),5,statStepSec);
      if (isempty(mc))
        mc=mc1;
        sc=sc1;
        %dc=dc1;
      else
        mc=mc+mc1;
        sc=sc+sc1;
        %dc=dc+dc1;
      end
    end  
    %dc=dc/length(model);
  else
    %dc=diffCurve(zscore(ft(:,selF),0,1),5,statStepSec);
    
    [ismusic,isspeech]=getCat(outputs);
    mc=max(allC(:,ismusic),[],2);
    sc=max(allC(:,isspeech),[],2);
  end
  mc=mc./(mc+sc+eps);
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

function dc=diffCurve(feat,winSec,statStepSec)
  dc=nan(size(feat,1),1);
  ws=round(winSec/statStepSec);
  for i=ws+1:size(feat,1)-ws+1
    dc(i)=cosineSim(feat(i-ws:i-1,:),feat(i:i+ws-1,:));
  end
end

function kl=klCurve(x,winSec,statStepSec)
  kl=nan(size(x,1),1);
  ws=round(winSec/statStepSec);
  for i=ws+1:size(x,1)-ws+1
    kl(i)=klSDiv(x(i-ws:i-1,:),x(i:i+ws-1,:));
  end
end

function [bound,smbT,smbV]=readBoundaries(name,stepSec)
  f1=fopen(name);
  C = textscan(f1,'%f,%f,%c');
  fclose(f1);
  
  smbT=0:stepSec:max(C{1}+C{2});
  msVal=zeros(length(smbT),2);
  for j=1:length(C{1})
    if (C{3}(j)=='s')
      msVal(smbT>=C{1}(j) & smbT<=C{1}(j)+C{2}(j),2)=1;
    else
      msVal(smbT>=C{1}(j) & smbT<=C{1}(j)+C{2}(j),1)=0.5;
    end
  end
  smbV=sum(msVal,2);
  bound=find(diff([2;smbV;2])~=0);
  bound=[(bound(1:end-1)-1)*stepSec (bound(2:end)-bound(1:end-1))*stepSec smbV(bound(1:end-1))];
end

function kl = klSDiv(X, Y, asVectors)
  if nargin<3
    asVectors=0;
  end
  if (size(X,2)==1 || asVectors~=0)
    
    md=mean(X,1)-mean(Y,1);
    P=max(var(X,0,1),0.01);
    Q=max(var(Y,0,1),0.01);

    IQ=1./Q;
    R = P.*IQ;

    IP=1./P;
    R1 = Q.*IP;

    kl = 0.5*(R + R1 -2 + md.*(IQ+IP).*md);
    kl=mean(kl);
  else
    md=mean(X)-mean(Y);
    P=max(cov(X),0.01);
    Q=max(cov(Y),0.01);
    if (cond(Q)>10)
        IQ=pinv(Q);
    else
        IQ=inv(Q);
    end
    R = P*IQ;

    if (cond(P)>10)
        IP=pinv(P);
    else
        IP=inv(P);
    end
    R1 = Q*IP;
    kl = 0.5*(trace(R) + trace(R1) -2 + md*(IQ+IP)*md');

  end
end