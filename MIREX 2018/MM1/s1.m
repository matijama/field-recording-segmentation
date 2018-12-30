function res = s1(inp,rowcol)
% res = s1(inp,rowcol) scales the input vector, so that its maximum positive value is one
% if inp is a matrix, then rowcol==0 scales the entire matrix to one, rowcol==1 scales
% each row to one and rowcol==2 scales each column to one. Default rowcol=0;

if (nargin==2)
   rc=rowcol;
else
   rc=0;
end

res=zeros(size(inp));

if rc==0
   if (max(inp(:))~=0)
      res=inp/max(inp(:));
   end
elseif rc==1
   m=max(inp,[],2);
   for i=1:size(inp,1)
      if (m(i)~=0)
         res(i,:)=inp(i,:)/m(i);
      end
   end
else 
   m=max(inp);
   for i=1:size(inp,2)
      if (m(i)~=0)
         res(:,i)=inp(:,i)/m(i);
      end
   end
end
         

