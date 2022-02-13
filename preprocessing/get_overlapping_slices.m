
function segments = get_overlapping_slices(spect_dB, example_overlap_fraction, dwell_time, duration_s)
    [nrows_in_spectrogram, ncols_in_spectrogram] = size(spect_dB);
    
    frames_in_dwell_time = round(dwell_time*ncols_in_spectrogram/duration_s);
    num_of_overlapping_points = fix(example_overlap_fraction * frames_in_dwell_time);

    segments =  struct('Data', {});
    start_column = 1; % Starting column in an image
    stop_column = start_column + frames_in_dwell_time - 1; % Stopping column in an image
    segment_number = 1;
    
    while stop_column < ncols_in_spectrogram % Check if there are still available points
       segments(segment_number).Data = spect_dB(:,start_column:stop_column);
       start_column = stop_column - num_of_overlapping_points + 1;
       stop_column = start_column + frames_in_dwell_time - 1;
       segment_number = segement_number + 1;
    end
end
 



