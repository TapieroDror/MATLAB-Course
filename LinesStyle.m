% MATLAB course for electrical engineering students - class3
% Class demonstration
% Plot location of walker for 4 realizations
clear all;close all;clc;
Nsteps = 20;
n = 0:Nsteps;
hold on
plot(n,RW1(Nsteps,-1,1),'x')  
plot(n,RW1(Nsteps,-1,1),'--g')  
plot(n,RW1(Nsteps,-1,1),'.-')  
plot(n,RW1(Nsteps,-1,1),'-.ko')  