clear;

% Load parameters in
params;

% Define storage structure
processed_data_struct =  struct('Data', {}, 'Label', {}); 

% Get file names
trk_data_struct_files = dir(fullfile(data_dir,'*.mat')); 

tic % start timer
for iFile = 1:length(trk_data_struct_files)
    file_name = trk_data_struct_files(iFile).name; %file_name = trk_data_struct_files(60).name;
    trk_data_struct = load_trk_data(data_dir, file_name);
    stft_data_struct = get_stft_data(trk_data_struct, window_length, overlap_fraction, fft_length, filter_params);
    examples_struct = generate_processed_examples(stft_data_struct, dwell_time, example_overlap_fraction);
    processed_data_struct = [processed_data_struct, examples_struct];
end
toc % stop_timer

save('Data\Processed\havsdata.mat', 'processed_data_struct', '-v7.3');

%imagesc(processed_data_struct(12).Data);colorbar;




