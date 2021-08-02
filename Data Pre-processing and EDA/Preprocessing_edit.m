clear;

% Load parameters
params;

% Filter
filter_props = fdesign.notch('N,F0,BW,Ap', N, F0, BW, Ap);
filter_params = design(filter_props);

HAVSDatasetStruct =  struct('Data', {}, 'Label', {});    
trk_data_files = dir(fullfile(data_dir,'*.mat'));

tic % start timer
for iFile = 1:length(trk_data_files)
    file_name = trk_data_files(iFile).name;
    file_path = strcat(data_dir, file_name);
    trk_data_struct = load(file_path);
    trk_data = trk_data_struct.trkdata;
    spectrograms_struct = generate_spectrograms(trk_data, window_length, overlap_fraction, fft_length, filter_params);
    examples_struct = generate_processed_examples(spectrograms_struct, dwell_time, example_overlap_fraction);
    HAVSDatasetStruct = [HAVSDatasetStruct, examples_struct];
end
toc % stop_timer

save('havsdata', 'HAVSDatasetStruct', '-v7.3');

%imagesc(HAVSDatasetStruct(12).Data);colorbar;




