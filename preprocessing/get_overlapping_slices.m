
function spect_slices = get_overlapping_slices(spect_dB, exampleoverlap_fraction, dwell_time, duration_s)
    [nrows, ncols] = size(spect_dB);
    
    frames_in_spec = round(dwell_time*ncols/duration_s);
    numOfOverlappingPoints = fix(exampleoverlap_fraction * frames_in_spec);

    spect_slices =  struct('Data', {});
    startCol = 1;
    for slice_num=1:ncols
       stopCol = startCol + frames_in_spec -1;
       if(stopCol < ncols) % Check if there are available points
           spect_slices(slice_num).Data = spect_dB(:,startCol:stopCol);
           startCol = stopCol - numOfOverlappingPoints + 1;
       end
    end
end
 



