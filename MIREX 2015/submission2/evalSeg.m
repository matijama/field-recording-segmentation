function [res,bound,boundG]=evalSeg(inFile, windowSec)

  if nargin<2
    windowSec=1;
  end
  [dName,name,suffix]=fileparts(inFile);
  [boundG,smbTG,smbVG]=readBoundaries(fullfile(dName,[name '.csv']),',',0.05);
  [bound,smbT,smbV]=readBoundaries(fullfile(dName,[name suffix '.seg']),'\t',0.05);
  t=find(smbV>1.1); % set overlapping to next value
  smbV(t)=smbV(t+1);
  
  
  smbVG(end+1:length(smbV))=0;
  smbV(end+1:length(smbVG))=0;
  
  tp=nnz(smbVG==smbV & smbVG>0 & smbVG<1.1) + nnz(smbV>0 & smbVG>=1.1);
  prec=tp/nnz(smbV>0);
  rec=tp/(nnz(smbVG>0 & smbVG<1.1) + 2*nnz(smbVG>=1.1));
  f1=2*prec*rec/(prec+rec+eps);
  
%   a ground truth segment is assumed to be correctly detected if the system identifies the right class (Music/Speech) 
%     AND the detected segment`s onset is within a 1000ms range( +/- 500ms) of the onset 
%   (onset-offset) a ground truth segment is assumed to be correctly detected if the system identifies the right class (Music/Speech), 
%     AND the detected segment`s onset is within +/- 500ms of the onset of the ground truth segment, 
%     AND the detected segment's offset is EITHER within +/- 500ms of the offset of the ground truth segment OR within 20% of the ground truth segment's length.
  tpo=0; tpoo=0;
  for i=1:size(boundG,1)
    t=find(abs(bound(:,1)-boundG(i,1))<=windowSec & abs(bound(:,3)-boundG(i,3))<0.1);
    if (any(t))
      tpo=tpo+1;
      if (any(abs(bound(t,1)+bound(t,2)-boundG(i,1)-boundG(i,2))<=windowSec | min(bound(t,2),boundG(i,2))./max(bound(t,2),boundG(i,2))>=0.8  ))
        tpoo=tpoo+1;
      end
    end
  end
  pr=[tpo/size(bound,1) tpo/size(boundG,1) tpoo/size(bound,1) tpoo/size(boundG,1)];
  res=[prec rec f1 pr(1:2) 2*pr(1)*pr(2)/(pr(1)+pr(2)+eps) pr(3:4) 2*pr(3)*pr(4)/(pr(3)+pr(4)+eps)];
  
end

function [bound,smbT,smbV]=readBoundaries(name,sep,stepSec)
  f1=fopen(name);
  C = textscan(f1,['%f' sep '%f' sep '%c']);
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
  %bound=[(bound(1:end-1)-1)*stepSec (bound(2:end)-bound(1:end-1))*stepSec smbV(bound(1:end-1))];
  bound=[C{1} C{2} 1-(C{3}~='s')*0.5];
end
