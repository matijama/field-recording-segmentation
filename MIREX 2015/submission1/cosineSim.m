function x=cosineSim(a,b)
%function x=cosineSim(a,b)

x=(a(:)'*b(:))/sqrt((a(:)'*a(:))*(b(:)'*b(:)));

