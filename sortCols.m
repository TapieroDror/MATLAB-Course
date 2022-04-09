%A function that receives a matrix and arranges its organs by rows
function Msort = sortCols(M)
Msort=sort(M(:));
Msort=reshape(Msort,size(M));
