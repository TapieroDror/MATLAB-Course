function [y] = Step2stOrder(k_s,omega_n,zeta,time_array)
% MATLAB course for electrical engineering students - class 3
% Class demonstration
% construct a second order Step siglal.
w_d=omega_n*sqrt(1-zeta^2);
y=k_s*(1-exp(-zeta*omega_n*time_array).*(cos(w_d*time_array)+zeta/(1-zeta^2)*sin(w_d*time_array)));
