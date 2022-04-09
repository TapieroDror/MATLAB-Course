function NoisedSignal=Noise(Signal,SNRlinear)
%MATLAB course for engineering studens - class 5 
%Class demonstration
%add to Signal noise with a given SNR
Asignal=max(Signal)-min(Signal); %2 X Amplitude of the signal
Anoise=Asignal/SNRlinear; %2 X noise amplitude
Noise=Rand(size(Signal),-Anoise/2,Anoise/2);
NoisedSignal=Signal+Noise;
