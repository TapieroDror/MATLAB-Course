%A function that receives a matrix and arranges its organs by columns
function Msort = sortRows(M)
Msort=sort(M(:));
Msort=reshape(Msort,size(M));
Msort=Msort';