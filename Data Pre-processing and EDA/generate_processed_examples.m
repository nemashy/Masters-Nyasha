
function examplesStruct = generate_processed_examples(spectrogramsStruct, dwell_time, example_overlap_fraction)

    examplesStruct =  struct('Data', {}, 'Label', {}); % Creating a table to store results
    
    for posSpectrogram=1:length(spectrogramsStruct)
        
        spectrogramLabel = spectrogramsStruct(posSpectrogram).Label;
        spect_dB = spectrogramsStruct(posSpectrogram).Data;
        spectrogramDuration = spectrogramsStruct(posSpectrogram).DurationSeconds;
        if(spectrogramDuration > dwell_time) % Check if example duration is greater than dwell time
            %spect_slices = getSlices(spect_dB, num_slicesPerSpectrogram, dwell_time, spectrogramDuration);
            spect_slices = get_overlapping_slices(spect_dB, example_overlap_fraction, dwell_time, spectrogramDuration);
            % Add slices to the examples stucture
            for iSlice=1:length(spect_slices)
                examplesStruct(end+1).Data = spect_slices(iSlice).Data;
                examplesStruct(end).Label = spectrogramLabel;
            end
        end
    end
end

