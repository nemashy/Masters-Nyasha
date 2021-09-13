clear;

% Load parameters
params;

[results, stft_data_struct] = measure_peak_snr(fft_lengths, window_lengths, overlap_fractions, data_dir, filter_params);

function [results, stft_data_struct] = measure_peak_snr(fft_lengths, window_lengths, overlap_fractions, data_dir, filter_params)

    max_num_experiments = length(fft_lengths)*length(window_lengths)*length(overlap_fractions);
    experiment_num = 1;
    data_files = dir(fullfile(data_dir,'*.mat'));
    results = struct('PSNR', cell(max_num_experiments, 1), 'FFTLength', cell(max_num_experiments, 1), 'WindowLength', cell(max_num_experiments, 1), 'OverlapFraction', cell(max_num_experiments, 1)); % Creating a table to store results
    for fft_length = window_lengths
        for window_length = window_lengths
            if(window_length <= fft_length)
                for overlap_fraction = overlap_fractions
                    for iFile = 54:54 %length(data_files)
                        file_name = data_files(iFile).name;
                        disp(file_name);
                        trk_data_struct = load_trk_data(data_dir, file_name);
                        stft_data_struct = get_stft_data(trk_data_struct, window_length, overlap_fraction, fft_length, filter_params);
                        psnr = calc_psnr(stft_data_struct);
                    end
                    % Save to results file
                    results(experiment_num).PSNR = psnr;
                    results(experiment_num).FFTLength = fft_length;
                    results(experiment_num).WindowLength = window_length;
                    results(experiment_num).OverlapFraction = overlap_fraction;
                    disp('Finished iteration number ' + string(experiment_num))
                    experiment_num = experiment_num + 1;
                end
            end
        end
    end
end

% Finds the average SNR of spectrograms in each trk_data_struct file
function psnr = calc_psnr(spectrograms)
    maxLevel_dB = 0;
    for posExample = 1:length(spectrograms)
        if(strcmp(spectrograms(posExample).Label, "clutter/noise"))
            disp(var(spectrograms(posExample).Data(:)));
            Noise_dB = 10*log10(var(spectrograms(posExample).Data(:)));
        else
            disp(spectrograms(posExample).Label);
            Current_Max_Level_dB = max(20*log10(abs(spectrograms(posExample).Data)), [], 'all');
            if(Current_Max_Level_dB > maxLevel_dB)
                maxLevel_dB = Current_Max_Level_dB;           
            end
        end    
    end
    psnr = maxLevel_dB - Noise_dB;
end
