function RandomNumbers=Rand(ArraySize,min,max)
% MATLAB course for engineering students - class 5
% Class demonstration
% construct a random array of constant distribution.
% ArraySize = size of random array
% min = lower edge of distribution
% max = upper edge of distribution
RandomNumbers=rand(ArraySize); % range from 0 to 1
RandomNumbers=RandomNumbers*(max-min); % range from 0 to (max-min)
RandomNumbers=RandomNumbers+min; % range from min to max