function ton=Tonality(w, sr, ceps)
% function ton=Tonality(w, sr, ceps)
% calculates Tonality as described in 
% NOISE ROBUST FEATURES FOR SPEECH/MUSIC DISCRIMINATION IN REAL-TIME TELECOMMUNICATION
% Tonality is an indication of how tonal/noisy the sound is (higher - more
% tonal)
% w and sr are the wav file and sample rate respectively
% cepst are optionally precalculated cepstral (not MEL!) coefficients

if nargin<2 sr=11025; end

winSize=0.05;
winSize=pow2(floor(log2(winSize*sr)));

%fr=round([400 3000]*winSize/sr);
fr=round(sr./[1000 90])+1;
if nargin<3
    step=winSize/2;
    
    ceps=cepst(w,sr,'M',fr(2),winSize,step);
    
end
ton=max(ceps(:,fr(1):fr(2)),[],2);

