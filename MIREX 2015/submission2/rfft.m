function y=rfft(x,n)
%RFFT     FFT of real data Y=(X,N)
% Data is truncated/padded to length N if specified.
%   N even:	(N+2)/2 points are returned with
% 			the first and last being real
%   N odd:	(N+1)/2 points are returned with the
% 			first being real



%      Copyright (C) Mike Brookes 1998
%
%      Last modified Fri Apr  3 14:57:19 1998
%
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
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

if nargin < 2
  y=fft(x);
else
  y=fft(x,n);
end
if size(y,1)==1
  m=length(y);
  y(floor((m+4)/2):m)=[];
else
  m=size(y,1);
  y(floor((m+4)/2):m,:)=[];
end

