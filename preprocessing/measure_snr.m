clear;

% Load parameters
params;

[results, stft_data_struct] = measure_peak_snr(window_lengths, overlap_fractions, data_dir, filter_params);

function [results, stft_data_struct] = measure_peak_snr(window_lengths, overlap_fractions, data_dir, filter_params)
    random_file_num = randsample(8, 5) + 61;
    max_num_experiments = length(window_lengths)*length(overlap_fractions);
    experiment_num = 1;
    time_stamps_AvgPSNR = [];
    data_files = dir(fullfile(data_dir,'*.mat'));
    results = struct('SNR', cell(max_num_experiments, 1), 'FFTLength', cell(max_num_experiments, 1), 'WindowLength', cell(max_num_experiments, 1), 'OverlapFraction', cell(max_num_experiments, 1)); % Creating a table to store results
        for window_length = window_lengths

                for overlap_fraction = overlap_fractions
                    for iFile = 1: length(random_file_num) %length(data_files)
                        disp(random_file_num(iFile));
                        file_name = data_files(random_file_num(iFile)).name;
                        disp(file_name);
                        disp(random_file_num);
                        trk_data_struct = load_trk_data(data_dir, file_name);
                        stft_data_struct = get_stft_data(trk_data_struct, window_length, overlap_fraction, window_length, filter_params);
                        SNR = calc_SNR(stft_data_struct);
                        time_stamps_AvgPSNR(end+1) = SNR;
                    end
                    
                    % Save to results file
                    results(experiment_num).SNR = mean(time_stamps_AvgPSNR);
                    results(experiment_num).FFTLength = window_length;
                    results(experiment_num).WindowLength = window_length;
                    results(experiment_num).OverlapFraction = overlap_fraction;
                    disp('Finished iteration number ' + string(experiment_num))
                    experiment_num = experiment_num + 1;
                end

        end
end

% Finds the average SNR of spectrograms in each trk_data_struct file
function AvgPSNR = calc_SNR(spectrograms)
    maxLevel_dB = 0;
    avg_peak_value_dB = [];
    for posExample = 1:length(spectrograms)
        if(strcmp(spectrograms(posExample).Label, "clutter/noise"))
            histogram(imag(spectrograms(posExample).Data(:)));
            xlim([-100000, 100000]);
            histogram(real(spectrograms(posExample).Data(:)));
            xlim([-100000, 100000]);
            Noise_dB = 10*log10(var(spectrograms(posExample).Data(:)));
        else
            disp(spectrograms(posExample).Label);
            peak_level_in_each_column = max(20*log10(abs(spectrograms(posExample).Data)), [], 'all');
            
            avg_peak_value_dB(end+1) = mean(peak_level_in_each_column);

        end    
    end
    AvgPSNR = mean(avg_peak_value_dB - Noise_dB);
end
