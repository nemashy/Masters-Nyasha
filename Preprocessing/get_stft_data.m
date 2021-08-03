
function spectograms_struct = generate_spectrograms(trkdata, window_length, overlap_fraction, fftLength, filter_params)
    if isa(trkdata, 'struct') % Check if the datatype is a 'struct'
        spectograms_struct = struct('Data',cell(length(trkdata), 1), 'Label', cell(length(trkdata), 1), 'DurationSeconds', cell(length(trkdata), 1));
        for range_bin_pos = 1:length(trkdata) % range_bin_pos = 16
            fs_Hz = trkdata(range_bin_pos).PRF; % sampling frequency = PRF
            [IQ_samples, label] = get_IQ_samples(range_bin_pos, trkdata);
            if(nargin == 5) % If there is a filter
                IQ_samples = filter(filter_params,IQ_samples);
            end
            % save('nyasha_signal.mat','IQ_samples', 'label');
            window = hamming(window_length);
            overlap_length = overlap_fraction * window_length;
            % Get STFT of the example
            [S, ~, T] = stft(IQ_samples,fs_Hz,'Window',window,'OverlapLength',overlap_length,'FFTLength',fftLength);
            spectograms_struct(range_bin_pos).Data = S;
            spectograms_struct(range_bin_pos).Label = label;
            spectograms_struct(range_bin_pos).DurationSeconds = T(end);
        end
    else
        disp('Needs a struct data type');
    end
end

function [IQ_samples, label] = get_IQ_samples(range_bin_pos, trkdata)
    I_data = double(trkdata(range_bin_pos).trk_data_real);
    Q_data = double(1i*trkdata(range_bin_pos).trk_data_imag);
    IQ_samples = I_data + Q_data;
    label = char(trkdata(range_bin_pos).class);
end
    
