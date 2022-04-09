function R = RW1(N,a,b)
% MATLAB course for electrical students class 5
% Class demonstration
% create a 1D Random Walk of N steps
% with displacements between a,b
u = Rand([N 1],a,b); % produce displacements
R = cumsum(u); % locations after each step
R = [0 ; R];


