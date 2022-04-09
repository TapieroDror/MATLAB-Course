function xyz_cumsum = RWgrid(N,s)
% MATLAB course for electrical engineering students
% Class demonstration
% create a Random Walk of N steps, each step randomly choosing from s.
% s is a matrix of vectors nxd, where n is the number of vectors and d is the dimensionality of the walk
% Assume starting point at the origin.
n = size(s,1); % number of possible steps {number of rows only}
indices = randi(n,N,1); % array of indices of chosen vectors for each step.
steps = s(indices,:); % 2 columns of steps
changes = [zeros(1,size(s,2)) ; steps]; % 1st location is the initial point
xyz_cumsum = cumsum(changes);

