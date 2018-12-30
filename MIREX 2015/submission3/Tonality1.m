function ton=Tonality1(w, sr, ceps, ft)
% function ton=Tonality(w, sr, ceps, ft)
% calculates Tonality. % Tonality is an indication of how tonal/noisy the sound is (higher - more
% tonal)
% Modifies the algorithm described in 
% NOISE ROBUST FEATURES FOR SPEECH/MUSIC DISCRIMINATION IN REAL-TIME TELECOMMUNICATION
% by taking the max cepstrum and multiplying with the ration of energy of
% the first five partials of the found series against the entire spectrum; 
% this weighs down the noisy parts, which accidentally have high cepstrum
% values but no actual partials
%
% w and sr are the wav file and sample rate respectively
% cepst are optionally precalculated cepstral (not MEL!) coefficients

if nargin<2 
    sr=11025; 
end

winSize=0.05;
winSize=pow2(floor(log2(winSize*sr)));

%fr=round([400 3000]*winSize/sr);
fr=round(sr./[1000 90])+1;
if nargin<3
    step=winSize/2;
    
    [ceps, ft]=cepst(w,sr,'M0S',fr(2),winSize,step,winSize);
    
end

% this gets to be modified
%ton=max(ceps(:,fr(1):fr(2)),[],2);

ft=abs(ft).^2; 
bins=(size(ft,2)-1)*2;
[~, loc]=max(ceps(:,fr(1):fr(2)),[],2);
loc=loc+fr(1)-2; % 0th is missing, so has to be +1
t=round([1:5]'*(bins./loc'))+1;
t=min(numel(ft),bsxfun(@plus,1:size(ft,1),(t-1)*size(ft,1)));
%ton=ton.*sum(ft(t))'./sum(ft,2);
a=sort(ft,2,'descend');
ton=sum(ft(t))'./sum(a(:,1:5),2);
