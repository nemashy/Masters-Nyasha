% Signal to noise ratio
clear all;
close all;

% User input parameters
N = 256;
fs = 500;
f = 93.75;  % Corresponds exactly to a frequency bin 
            % Explore what happens when f is not aligned to a frequency bin
SNR_dB = 100; 
A_n = 10^(-SNR_dB/20);  

% Compute signal and noise in the time-domain 
t = (0:1:(N-1))*1/fs;
Signal = exp(1i*2*pi*f*t);
Noise = A_n*(normrnd(0,1,1,N) + 1i*normrnd(0,1,1,N))*1/sqrt(2); % 1/sqrt(2) used to normalise the noise power to one 
SignalPlusNoise = Signal + Noise; 

% Verification - Estimate the SNR in the time-domain 
Estimate_SNR_TimeDomain = 10*log10(var(Signal)/var(Noise));
disp(['User set SNR =  ', num2str(SNR_dB), ' dB' ])
disp(['Estimate of SNR in the time domain =  ', num2str(Estimate_SNR_TimeDomain), ' dB' ])


% ------------------------------------------
% Plot signal in the time domain 
figure; plot(t, real(SignalPlusNoise), '-b');
hold on;
plot(t, real(Signal), '-r');
xlabel('Time (s)');
legend('Signal + Noise', 'Signal only');
grid on;
% ------------------------------------------

FFT_SignalOnly = fft(Signal);
FFT_SignalPlusNoise = fft(SignalPlusNoise);
FFT_Noise = fft(Noise);

FrequencyAxis_Hz = (-N/2:1:(N/2-1))*fs/N;

% -----------------------------------------------------------------------------------
% Plot signal + noise in the frequency domain 
FFT_SignalPlus_Noise_dB = 20*log10(abs(fftshift(FFT_SignalPlusNoise)));

figure; plot(FrequencyAxis_Hz, FFT_SignalPlus_Noise_dB);
xlabel('Frequency (Hz)');
grid on;
% -----------------------------------------------------------------------------------

% -----------------------------------------------------------------------------------
% Plot noise in the frequency domain 
histogram(real(FFT_Noise));
FFT_Noise_dB = 20*log10(abs(fftshift(FFT_Noise)));

figure; plot(FrequencyAxis_Hz, FFT_Noise_dB);
xlabel('Frequency (Hz)');
grid on;
% -------------------------------------------------------------------------------------------

% Verification - Estimate the SNR in the frequency domain 
Estimate_SNR_FreqDomain = 10*log10(var(FFT_SignalOnly)/var(FFT_Noise));
disp(['Estimate of SNR in the frequency domain =  ', num2str(Estimate_SNR_FreqDomain), ' dB' ])

% Nyasha to do: Estimate the SNR from SignalPlusNoise and Noise signals only 
% ----------------------------------------------------------------------------


% --------

% SNR_dB = 10*log10(var(fftshift(FFT_SignalPlusNoise))) - 10*log10(var(FFT_Noise)); % Book Equation 
SNR_dB = max(FFT_SignalPlus_Noise_dB) - 10*log10(var(FFT_Noise)) - 10*log10(N); % Equation without sources


Adjusted_SNR_dB = SNR_dB;  % Adjusting for the integration gain

disp(['Nyasha: Estimate of SNR in the frequency domain =  ', num2str(Adjusted_SNR_dB), ' dB' ])

% disp(['Estimate of max SNR in the frequency domain =  ', num2str(Estimate_MaxSNR_FreqDomain), ' dB' ])

