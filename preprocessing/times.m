clear;

params;

time_stamps_array = [];
classes_array = {};
data_files = dir(fullfile(data_dir,'*.mat'));
classes_array = [];
for iFile = 1:length(data_files)
    file_name = data_files(iFile).name;
    trk_data = load_trk_data(data_dir, file_name);
    
    for example_number = 1:length(trk_data)
        time_stamps_array = [time_stamps_array trk_data(example_number).measurement];
        classes_array{end+1} = trk_data(example_number).class{1};
    end
    disp(unique(classes_array));
end

save('time_stamps.mat', 'time_stamps_array', '-v7.3');
save('classes_array.mat', 'classes_array', '-v7.3');

unique_timesatmps_array = unique(time_stamps_array);
unique_timesatmps_array = unique_timesatmps_array(:);
% All the examples were taken within 74 unique timestamps