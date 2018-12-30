function y = percfilt1(x,p,n)
%percfilt1 One dimensional percentile filter
% derived from medfilt1 in sp toolbox

[x, nshifts] = shiftdim(x);

% Verify that the block size is valid.
siz = size(x);
blksz = siz(1); % siz(1) is the number of rows of x (default)

% Initialize y with the correct dimension
y = zeros(siz); 

% Call medfilt1D (vector)
for i = 1:prod(siz(2:end)),
	y(:,i) = percfilt1D(x(:,i),n,blksz,p);
end

y = shiftdim(y, -nshifts);


%-------------------------------------------------------------------
%                       Local Function
%-------------------------------------------------------------------
function y = percfilt1D(x,n,blksz,p)
%MEDFILT1D  One dimensional median filter.
%
% Inputs:
%   x     - vector
%   n     - order of the filter
%   blksz - block size

nx = length(x);
if rem(n,2)~=1    % n even
    m = n/2;
else
    m = (n-1)/2;
end
X = [zeros(m,1); x; zeros(m,1)];
y = zeros(nx,1);

% Work in chunks to save memory
indr = (0:n-1)';
indc = 1:nx;
for i=1:blksz:nx
    ind = indc(ones(1,n),i:min(i+blksz-1,nx)) + ...
          indr(:,ones(1,min(i+blksz-1,nx)-i+1));
    xx = reshape(X(ind),n,min(i+blksz-1,nx)-i+1);
    y(i:min(i+blksz-1,nx)) = prctile(xx,p,1);
end
