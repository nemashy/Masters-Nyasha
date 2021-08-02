clear;

% Load parameters in
params;

% Define storage structure
HAVSDatasetStruct =  struct('Data', {}, 'Label', {}); 

% Get file names
trk_data_files = dir(fullfile(data_dir,'*.mat')); 

tic % start timer
for iFile = 1:length(trk_data_files)
    file_name = trk_data_files(iFile).name; %file_name = trk_data_files(60).name;
    file_path = strcat(data_dir, file_name);
    trk_data_struct = load(file_path);
    trk_data = trk_data_struct.trkdata;
    spectrograms_struct = generate_spectrograms(trk_data, window_length, overlap_fraction, fft_length, filter_params);
    examples_struct = generate_processed_examples(spectrograms_struct, dwell_time, example_overlap_fraction);
    HAVSDatasetStruct = [HAVSDatasetStruct, examples_struct];
end
toc % stop_timer

save('Data\havsdata.mat', 'HAVSDatasetStruct', '-v7.3');

%imagesc(HAVSDatasetStruct(12).Data);colorbar;




