fs = 100;
N = 128;
FrequencyAxis_Hz = (-N/2:1:(N/2-1))*fs/N;

% -----------------------------------------------------------------------------------
% Plot signal + noise in the frequency domain 
FFT_SignalPlus_Noise_dB = 20*log10(abs((rectwin(128))));

figure; plot(FFT_SignalPlus_Noise_dB);
xlabel('Frequency (Hz)');
grid on;

figure;
L = 64;


H1 = hamming(64);
H2 = hanning(64);
H4 = blackman(64);
H5 = rectwin(64);
wvt = wvtool(H1, H2, H4, H5);
legend(wvt.CurrentAxes,'Hamming','Hanning', 'Kaiser', 'Blackman', 'Rectangular');