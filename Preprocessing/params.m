% Data folder path
data_dir = '.\Data\Unprocessed';

% SFTF parameters
dwell_time = 4;
fft_length = 128;
overlap_fraction = 0.5;
window_length = 128;
window = hamming(window_length);

% Spectrogram slicing parameters
slices_per_spec = 4; 
example_overlap_fraction = 0.25;

% Filter design
N = 6; % Filter order
F0 = 0; % Centre frequency
BW = 0.045; % Bandwidth
Ap = 0.5; % Passband ripple

filter_props = fdesign.notch('N,F0,BW,Ap', N, F0, BW, Ap);
filter_params = design(filter_props);

% Parameters tuning
fft_lengths = [64 128 256];
window_lengths = [32 64 128 256];
overlap_fractions = [0.25 0.50 0.75];

