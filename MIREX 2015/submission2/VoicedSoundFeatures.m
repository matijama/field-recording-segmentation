function [feat,featNames]=VoicedSoundFeatures(ton,et,statWinSize,statStep, stepSec)
% function [feat,featNames]=VoicedSoundFeatures(ton,et,statWinSize,statStep, stepSec)
% calculates features of voiced sounds as in "Discrimination of Speech and
% Monophonic Singing in Continuous Audio Streams"

%tonalityThreshold=40;
tonalityThreshold=0.4;

    seg=min(find(diff([0;1-et;0])),length(ton));
    seg=reshape(seg,2,length(seg)/2);
    segS=min(find(diff([0;et;0])),length(ton));
    segS=reshape(segS,2,length(segS)/2);

    soundVoice=arrayfun(@(x,y) (ton(x:y)'*hamming(y-x+1))/sum(hamming(y-x+1)), seg(1,:), seg(2,:));
    soundDur=(seg(2,:)-seg(1,:)+1)*stepSec;
    soundDurS=(segS(2,:)-segS(1,:)+1)*stepSec;

    feat=zeros(ceil(stepSec*length(ton)/statStep)+1,5);
    featNames={'voicedRate','Silence_mean','Silence_std','Silence_var','Voiced_mean','Voiced_std','Voiced_var'};
    
    j=1;
    idx=1;
    ton(et~=0)=0; % tonality under energy threshold==0
    while idx<=length(ton)
      rng=max(1, idx-floor(statWinSize/2)):min(idx+floor(statWinSize/2),length(ton));
      
%        feat(j,1)=nnz(ton(idx:idx+statWinSize-1,:)>tonalityThreshold)/statWinSize;        
        feat(j,1)=nnz(ton(rng,:)>tonalityThreshold)/(sum(~et(rng))+eps);
        t=find(seg(:)>rng(1) & seg(:)<rng(end));
        if (any(t))
            t=floor((t-1)/2+1);            
            td=soundDur(t);
            td=td(soundVoice(t)>tonalityThreshold);
            if any(td)
                feat(j,5:7)=[mean(td) std(td) std(td).^2];                
            else
                feat(j,5:7)=[0 0.5 0.25];                
            end
            tS=find(segS(:)>rng(1) & segS(:)<rng(end));
            if (any(tS))
                tS=floor((tS-1)/2+1);            
                feat(j,2:4)=[mean(soundDurS(tS)) std(soundDurS(tS)) std(soundDurS(tS)).^2];
            else
                feat(j,2:4)=[0 0.5 0.25];
            end
        else
            feat(j,2:7)=[0 0.5 0.25 0 0.5 0.25];
        end
        
        idx=round(j*statStep/stepSec+1);
        j=j+1;
    end
    feat(j:end,:)=[];
