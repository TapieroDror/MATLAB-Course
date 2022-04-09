function [y] = Impulse1stOrder(k_s,tau,time_array)
% MATLAB course for electrical engineering students - class 3
% Class demonstration
% construct a first order Impulse siglal.
y=k_s/tau*exp(-time_array./tau);
