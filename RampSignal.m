function Signal=RampSignal(Amplitude,Tperiod,timeArray)
% MATLAB course for engineering students - class 6
% Class demonstration
% construct a "saw-tooth" signal.
% Amplitude = amplitude
% Tperiod = time (period) [constant]
% timeArray = an array of times t
sUnNormalized=mod(timeArray,Tperiod); %range from 0 to T
sNormalized=sUnNormalized/Tperiod; %range from 0 to 1
Signal=sNormalized*Amplitude; %range from 0 to A {FINEL SIGNAL}
