function [y] = Step1stOrder(k_s,tau,time_array)
% MATLAB course for electrical engineering students - class 3
% Class demonstration
% construct a first order Step siglal.
y=k_s*(1-exp(-time_array./tau));