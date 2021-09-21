
function examplesStruct = generate_processed_examples(spectrogramsStruct, dwell_time, example_overlap_fraction)

    examplesStruct =  struct('Data', {}, 'Label', {}, 'TimeStamp', {}); % Creating a table to store results
    
    for posSpectrogram=1:length(spectrogramsStruct)
        spectrogramLabel = spectrogramsStruct(posSpectrogram).Label;
        spectrogramTimestamp = spectrogramsStruct(posSpectrogram).TimeStamp;
        stft_data = spectrogramsStruct(posSpectrogram).Data;
        stft_data_dB = 20*log10(abs(stft_data));
        spectrogramDuration = spectrogramsStruct(posSpectrogram).DurationSeconds;
        if(spectrogramDuration > dwell_time) % Check if example duration is greater than dwell time
            %spect_slices = getSlices(stft_data_dB, num_slicesPerSpectrogram, dwell_time, spectrogramDuration);
            %spect_slices = get_overlapping_slices(stft_data_dB, example_overlap_fraction, dwell_time, spectrogramDuration);
            spect_slices = get_random_slices(stft_data_dB, dwell_time, spectrogramDuration);
            % Add slices to the examples stucture
            for iSlice=1:length(spect_slices)
                examplesStruct(end+1).Data = spect_slices(iSlice).Data;
                examplesStruct(end).Label = spectrogramLabel;
                examplesStruct(end).TimeStamp = spectrogramTimestamp;      
            end
        end
    end
end

