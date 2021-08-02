% Data folder path
data_dir = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';

% SFTF parameters
dwell_time = 4;
fft_length = 128;
overlap_fraction = 0.5;
window_length = 128;
window = hamming(window_length);

% Spectrogram slicing parameters
slices_per_spec = 4; 
example_overlap_fraction = 0.25;

% Filter params
N = 6; % Filter order
F0 = 0; % Centre frequency
BW = 0.045; % Bandwidth
Ap = 0.5; % Passband ripple

% Parameters tuning
fftLengths = [64 128 256];
windowLengths = [32 64 128 256];
overlapFractions = [0.25 0.50 0.75];

