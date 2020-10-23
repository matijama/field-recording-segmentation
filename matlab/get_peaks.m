function ret = get_peaks(q,tresh,keep)  
%get_peaks(x,tresh) returns peaks of vector x, that are larger then treshold value. Default treshold
%   is max(q)/20. Result contains peak values at peaks and zeros elsewhere. If x is a matrix,
%	 get peaks returns peaks for each row.
% keep means that peaks values are kept from one peak to the next

if nargin<3
    keep=0;
end
if (size(q,2)==1)
    q=q';
end

if nargin<2 | tresh==0
  tresh=0;
  trsh=0;
end

ret=zeros(size(q));
for jj=1:size(q,1)
	w=[q(jj,:) q(jj,end)];
	k=0;
	l=0;
   rise=0;
   prev=0;
   if tresh~=0
      if tresh>0 
          trsh=max(w)/tresh;
      else
          trsh=-tresh;
      end
   end
	for i=1:length(w)-1
  		if w(i)<w(i+1)
           if k~=0
              k=0;
           end
           w(i)=0;
           rise=1;
           if keep==1
               ret(jj,i)=prev;
           end
      elseif w(i)==w(i+1)
           rise=1;
           if (k==0)
              k=i;
           end;
           if keep==1
               ret(jj,i)=prev;
           end
           l=i+1;
      else
           if k~=0 
             if rise~=0
                 for ii=k:l
                    if ii==floor((k+l)/2)
                       ret(jj,ii)=w(ii)*(w(ii)>trsh);
                       prev=w(ii)*(w(ii)>trsh);
                   elseif keep==1
                       ret(jj,ii)=prev;
                   end
                 end
                 rise=0;
             elseif keep==1
                 ret(jj,i)=prev;
             end
              k=0;
           end
           if (rise==1)
              rise=0;
              ret(jj,i)=w(i)*(w(i)>trsh);
              prev=ret(jj,i);
           else
              w(i)=0;
              if keep==1
                  ret(jj,i)=prev;
              end
           end
     end
  end
     
  if k~=0 
     if rise~=0
        for ii=k:l
           if ii==floor((k+l)/2)
              ret(jj,ii)=w(ii)*(w(ii)>trsh);
              prev=ret(jj,ii);
          elseif keep==1
              ret(jj,ii)=prev;
          end
        end
     end
  end
end          
