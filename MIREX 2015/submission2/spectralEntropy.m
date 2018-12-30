function y=spectralEntropy(spectrum, srate, minF, maxF)
% function y=spectralEntropy(spectrum, srate, minF, maxF)
%
% Calculates the spectral entropy of the signal, as described by pikrakis
% spectrum is the STFT spectrum of the signal, already split in half (as returned i.e. by rfft)
%       (:,1) is time, (:,2) is freq
% srate is the sample rate (telling the max STFT frequency) 
% minF is the minimal frequency of filters for entropy calculation, maxF is the maximal Frequency of filters

%% get triangular (not specified in paper! ) filter coefficients as in Pampalk's ma_mfcc

freqs=minF*2.^((-1:floor(12*log2(maxF/minF))+1)/12);
nFreqs=length(freqs)-2;

filterWeights = zeros(nFreqs,size(spectrum,2));
triangleHeight = 2./(freqs(3:end)-freqs(1:end-2));
fft_freq = linspace(0,srate/2,size(spectrum,2));

for i=1:nFreqs,
    filterWeights(i,:) = ...
        (fft_freq > freqs(i) & fft_freq <= freqs(i+1)).* ...
        triangleHeight(i).*(fft_freq-freqs(i))/(freqs(i+1)-freqs(i)) + ...
        (fft_freq > freqs(i+1) & fft_freq < freqs(i+2)).* ...
        triangleHeight(i).*(freqs(i+2)-fft_freq)/(freqs(i+2)-freqs(i+1));
end

filteredSpectrum=filterWeights*(spectrum.*conj(spectrum))';
t=sum(filteredSpectrum); t(t==0)=eps;
filteredSpectrum=filteredSpectrum./repmat(t,size(filteredSpectrum,1),1);
filteredSpectrum(filteredSpectrum==0)=eps;
y=-sum(filteredSpectrum.*log2(filteredSpectrum));
