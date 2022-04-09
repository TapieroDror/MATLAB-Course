function sum = sum_var(varargin)
% this function sum all the arguments that the user put in,
% this function recived only 1X1 doubel matrix
sum=0;
fprintf('nargin: %d\n',nargin)
varargin
for i=1:nargin
    sum=sum+varargin{i};
end
end