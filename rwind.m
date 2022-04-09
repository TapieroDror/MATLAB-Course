%replace what is not divisible with d
%M - the original matrix
%d - The number you want to see which numbers are divided with it
%r - The number with which all numbers that are not divided with d will be
%replaced with it
function mat=rwind(M,d,r)
b=mod(M,d);
index=find(b>0);
M(index)=r;
mat=M;