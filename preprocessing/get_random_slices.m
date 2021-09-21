function spect_slices = get_random_slices(spect_dB, dwell_time, duration_s)

   
    spect_slices =  struct('Data', {});    
    [nrows, ncols] = size(spect_dB);

    num_frames = fix(dwell_time*ncols/duration_s);
    num_slices = fix(ncols/num_frames)

    for slice_num=1:num_slices
       start_col = randi([1, ncols - num_frames], 1);
       stop_col = start_col + num_frames;
       % Save data
       spect_slices(slice_num).Data = spect_dB(:,start_col:stop_col);
    end
end
