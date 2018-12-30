function [cc,values]=melcepst(s,fs,w,nc,p,n,inc,fl,fh,calcAlso)
% function [cc,values]=melcepst(s,fs,w,nc,p,n,inc,fl,fh)
%   MELCEPST Calculate the mel cepstrum of a signal 
%   Also optionally calculates spectral entropy, not related to melcepst, but used for session
%   classification so that we use fft information efficiently and not bump into memory errors
%   returns 
%           cc as coefficients, 
%           values contains additional values as specified by calcAlso (fft, entropy ...)
%
% from VOICEBOX
%
% Simple use: c=melcepst(s,fs)	% calculate mel cepstrum with 12 coefs, 256 sample frames
%				  c=melcepst(s,fs,'e0dD') % include log energy, 0th cepstral coef, delta and delta-delta coefs
%             return value f is the STFT of the signal
% Inputs:
%     s	 speech signal
%     fs  sample rate in Hz (default 11025)
%     nc  number of cepstral coefficients (default 12)
%     n   length of frame (default power of 2 <30 ms))
%     p   number of filters in filterbank (default floor(3*log(fs)) )
%     inc frame increment (default n/2)
%     fl  low end of the lowest filter as a fraction of fs (default = 0)
%     fh  high end of highest filter as a fraction of fs (default = 0.5)
%
%		w   any sensible combination of the following:
%
%				'R'  rectangular window in time domain
%				'N'	Hanning window in time domain
%				'M'	Hamming window in time domain (default)
%
%		      't'  triangular shaped filters in mel domain (default)
%		      'n'  hanning shaped filters in mel domain
%		      'm'  hamming shaped filters in mel domain
%
%				'p'	filters act in the power domain
%				'a'	filters act in the absolute magnitude domain (default)
%
%			   '0'  include 0'th order cepstral coefficient
%				'e'  include log energy
%				'd'	include delta coefficients (dc/dt)
%				'D'	include delta-delta coefficients (d^2c/dt^2)
%
%		      'z'  highest and lowest filters taper down to zero (default)
%		      'y'  lowest filter remains at 1 down to 0 frequency and
%			   	  highest filter remains at 1 up to nyquist freqency
%
%		       If 'ty' or 'ny' is specified, the total power in the fft is preserved.
%
%    calcAlso may include additional features to calculate/return as
%            'f'  fft 
%            'H' spectral entropy
%            'p' pitch density
%            'T' tonality
%            'O' tonality1
%            '4' 4Hz modulation
%            'c' real cepstrum (not mel, rdct(log(fft)))
%            'S' do pitch density and tonality on sqrt(abs(fft))
% 
% Outputs:	c     mel cepstrum output: one frame per row
%


%      Copyright (C) Mike Brookes 1997
%
%      Last modified Thu Jul 30 08:31:27 1998
%
%   VOICEBOX is a MATLAB toolbox for speech processing. Home page is at
%   http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
if nargin<2 fs=11025; end
if nargin<3 w='M'; end
if nargin<4 nc=12; end
if nargin<5 p=floor(3*log(fs)); end
if nargin<6 n=pow2(floor(log2(0.03*fs))); end
if nargin<9
   fh=0.5;   
   if nargin<8
     fl=0;
     if nargin<7
        inc=floor(n/2);
     end
  end
end

if nargin<10
    calcAlso='';
end

frameStep=1000;

ff=[]; 
ceps=[];
entropy=[];
tonality=[];
tonality1=[];
pitchDensity=[];
fourHzModulation=[];

numFrames=ceil((length(s)-n+1)/inc);
cc=zeros(numFrames,nc);
if (nargout>1)
    if any(calcAlso=='f')
        ff=zeros(numFrames,floor((n+2)/2));
    end
    if any(calcAlso=='c')
        ceps=zeros(numFrames,floor((n+2)/2));
    end
    if (any(calcAlso=='H'))
        entropy=zeros(numFrames,1);
    end
    if (any(calcAlso=='p'))
        pitchDensity=zeros(numFrames,1);
    end
    if (any(calcAlso=='T'))
        tonality=zeros(numFrames,1);
    end
    if (any(calcAlso=='O'))
        tonality1=zeros(numFrames,1);
    end
    if (any(calcAlso=='4'))
        fourHzModulation=zeros(numFrames,1);
    end
end

for currF=1:frameStep:numFrames
    startEnd=[(currF-1)*inc+1 min(length(s),(currF+frameStep-1-1)*inc+n)];
    if (isinteger(s))
        ss=double(s(startEnd(1):startEnd(2)))/32768;
    else
        ss=s(startEnd(1):startEnd(2));
    end
    if any(w=='R')
       z=enframe(ss,n,inc);
       nconst=n;
    elseif any (w=='N')
       z=enframe(ss,hanning(n),inc);
       nconst=sum(hanning(n));
    else
       z=enframe(ss,hamming(n),inc);
       nconst=sum(hamming(n));
    end
    f=rfft(z.')/nconst*2;
    ath=1/65536;

    if (~isempty(pitchDensity) || ~isempty(tonality) || ~isempty(tonality1) || ~isempty(ceps))
        % calc cepstrum (non-mel)
        t=max(abs(f),ath);
        if (any(calcAlso=='S'))
            y=sqrt(t);
        else
            y=log(t);
        end
        c=0.5*rdct(y).';
        nf=size(c,1);
        if (~isempty(ceps))
            ceps(currF:currF+nf-1,:)=c;
        end
        if (~isempty(pitchDensity))
            pitchDensity(currF:currF+nf-1)=PitchDensity([],fs,c);
        end
        if (~isempty(tonality))
            tonality(currF:currF+nf-1)=Tonality([],fs,c);
        end
        if (~isempty(tonality1))
            tonality1(currF:currF+nf-1)=Tonality1([],fs,c,t');
        end
    end

    [m,a,b]=melbankm(p,n,fs,fl,fh,w);
    m=bsxfun(@rdivide,m,sum(m>0,2)); % normalize
    if (any(w=='p') || any(w=='e'))
        pw=f(a:b,:).*conj(f(a:b,:));
    end
    if any(w=='p')
       pth=max(pw(:))*1E-6;
       t=max(m*pw,pth);
       y=log(t);
    else
       t=bsxfun(@max, m*abs(f(a:b,:)),full(ath*max(m,[],2)));
       y=log(t);
    end
    c=rdct(y).';
    if ~any(w=='0')
       c(:,1)=[];
    end
    if any(w=='e')
       c=[log(sum(pw)).' c];
    end
    
    nf=size(c,1);
    if (~isempty(ff))
        ff(currF:currF+nf-1,:)=f';
    end
    if (~isempty(entropy))
        entropy(currF:currF+nf-1)=spectralEntropy(f',11025,55,2200);
    end
    if (~isempty(fourHzModulation))
        if (currF==1)
            mcBuf=zeros(frameStep*2,size(t,1));       
            if (nf~=frameStep)
                mcBuf(frameStep+(1:frameStep),:)=[t'; repmat(mean(t,2)',frameStep-nf,1)];
            else
                mcBuf(frameStep+(1:frameStep),:)=t';
            end
            FHWinLen=0.5;
            FHWinLen=FHWinLen/(inc/fs);
            FHWinLenR=round(FHWinLen);
            FHt=cos(4*2*pi*linspace(0,FHWinLen,FHWinLenR));
        else
            mcBuf(1:frameStep,:)=mcBuf(frameStep+1:2*frameStep,:);
            if (nf~=frameStep)
                mcBuf(frameStep+(1:frameStep),:)=[t'; repmat(mean(t,2)',frameStep-nf,1)];
            else
                mcBuf(frameStep+(1:frameStep),:)=t';
            end
            for i=1:frameStep
                z=mcBuf(i:i+FHWinLenR-1,:);
                fourHzModulation(currF-frameStep+i-1)=sum(abs((FHt*z)./sum(z)));
            end            
        end
    end
    if p>nc
       c(:,nc+1:size(c,2))=[];
    elseif p<nc
       c=[c zeros(nf,nc-p)];
    end
    cc(currF:currF+nf-1,:)=c;  
end

if (~isempty(fourHzModulation))
    mcBuf(1:frameStep,:)=mcBuf(frameStep+1:2*frameStep,:);
    currF=currF+frameStep;
    for i=1:nf
        z=mcBuf(i:i+FHWinLenR-1,:);
        fourHzModulation(currF-frameStep+i-1)=sum(abs((FHt*z)./sum(z)));
    end            
end


nf=size(cc,1);
% calculate derivative

if any(w=='D')
  vf=(4:-1:-4)/60;
  af=(1:-1:-1)/2;
  ww=ones(5,1);
  cx=[cc(ww,:); cc; cc(nf*ww,:)];
  vx=reshape(filter(vf,1,cx(:)),nf+10,nc);
  vx(1:8,:)=[];
  ax=reshape(filter(af,1,vx(:)),nf+2,nc);
  ax(1:2,:)=[];
  vx([1 nf+2],:)=[];
  if any(w=='d')
     cc=[cc vx ax];
  else
     cc=[cc ax];
  end
elseif any(w=='d')
  vf=(4:-1:-4)/60;
  ww=ones(4,1);
  cx=[cc(ww,:); cc; cc(nf*ww,:)];
  vx=reshape(filter(vf,1,cx(:)),nf+8,nc);
  vx(1:8,:)=[];
  cc=[cc vx];
end
 
if nargout<1
   [nf,nc]=size(cc);
   t=((0:nf-1)*inc+(n-1)/2)/fs;
   ci=(1:nc)-any(w=='0')-any(w=='e');
   imh = imagesc(t,ci,cc.');
   axis('xy');
   xlabel('Time (s)');
   ylabel('Mel-cepstrum coefficient');
	map = (0:63)'/63;
	colormap([map map map]);
	colorbar;
elseif nargout>1
    values={};
    for i=1:length(calcAlso)
        switch calcAlso(i)
            case 'f'
                values{end+1}=ff;
            case 'c'
                values{end+1}=ceps;
            case 'H'
                values{end+1}=entropy;
            case 'T'
                values{end+1}=tonality;
            case 'O'
                values{end+1}=tonality1;
            case 'p'
                values{end+1}=pitchDensity;
            case '4'
                values{end+1}=fourHzModulation;
        end
    end                
end
