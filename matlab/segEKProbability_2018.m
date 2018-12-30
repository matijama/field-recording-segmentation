function [segs, eklT]=segEKProbability_2018(ec,kl,classTable,params,statStepInSec)
% function [segs, eklT]=segEKProbability_2018(ec,kl,classTable,params,statStepInSec)
% 
% segments the session according to ec - energy curve, kl - KL diff curve with a probabilistic approach
% returns the segment boundaries

  % for caching pdfs (speed)
  pdfDist=zeros(30000,size(classTable,2))-1;

  % segment lengths in seconds according to content type (gaussian means and stds)
  % solo, choir, instrumental, speech
  segLengthDistribution=[65.9401 107.2199 101.6937 140.3333;
                          52.2381 70.4295 99.0742 234.2829 ]; 
  segLengthDistribution=segLengthDistribution/statStepInSec;

  if nargin<4
      minLength=10/statStepInSec; % segments shorter than this are not allowed
      bLen=1;
  else
      minLength=params(3);
      bLen=params(5);
  end
    
  partialSum=zeros(size(classTable));
  partialSum(1,:)=sum(classTable(1:80,:));
  for i=2:size(classTable,1)-80+1
      partialSum(i,:)=partialSum(i-1,:)-classTable(i-1,:)+classTable(i+80-1,:);
  end

  % boundary candidates
  candidates = union(find(get_peaks(ec,0)),find(get_peaks(kl,0)));
  candidates=union(candidates,find(get_peaks(ec+kl,0)));
  if (isempty(candidates))
      segs=[1 length(ec)];
      eklT=[];
      return;
  end

  ec=ec(:); kl=kl(:);
  eklT=[ec(candidates) kl(candidates)];

  if (candidates(1)>1)
      candidates=[1 candidates];
      eklT=[0.5 0.5; eklT];
  end

  eklT=max(eklT,[],2);

  % limit prob to 0.01 & 1
  eklT = max(0.01,min(eklT,1-eps));

  cValPos=-log(eklT);
  cValNeg=-log(1-eklT);

  tbl=zeros(size(cValPos,1));
  bp=zeros(size(cValPos,1),1);

  tbl(1,1)=cValPos(1);

  % do DP 
  for i=2:size(tbl,1)
      for j=1:i-2 % (until i-1, all have no boundary at i-1);
          tbl(i,j)=tbl(i-1,j)+cValNeg(i-1);
      end       
      tbl(i,i)=inf;
      for j=1:i-1
         len=candidates(i)-candidates(j)+1;
         if (len>0 && pdfDist(len,1)==-1)
             pdfDist(len,:)=normpdfF(len*ones(1,size(classTable,2)),segLengthDistribution(1,:),segLengthDistribution(2,:),1);
         end            
         pLen=getSegLenProbability(i,j,candidates,classTable, partialSum, minLength, pdfDist(len,:));
         t=tbl(i,j)+tbl(j,j) + cValPos(i) + bLen*pLen;% + pConsist;% + pLen + pConsist;
         if (t<tbl(i,i))
             tbl(i,i)=t;
             bp(i)=j;
         end
      end
  end    
  
  % decode best path
  if (bp(end)~=0)
      bestPath=[bp(end) length(candidates)];
      while (bestPath(1)~=1)
          bestPath=[bp(bestPath(1)) bestPath];
      end

      segs=candidates(bestPath);
  else
      segs=[1 length(ec)];
  end
end


function p1 = getSegLenProbability(to,from,candidates,classTable, partialSum, minLength, pdfDist)
  % returns probability of segment length given parameters
    
  en=candidates(to);
  st=candidates(from);
  len=en-st;
  if (len<=minLength)
      p1=inf;
  else
      if (len>80)
          t=st:80:en;
          prof=sum(partialSum(t(1:end-1),:),1);
          prof=prof+sum(classTable(t(end):en,:),1);
          prof=prof/(len+1);
      else
          prof=mean(classTable(st:en,:),1);            
      end
      prof=prof/sum(prof);
      p1=-log(sum(pdfDist.*prof));        
  end
end

    