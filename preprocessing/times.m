clear;

radarDataFolderPath = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';

time_stamps_array = [];    
trackDataFiles = dir(fullfile(radarDataFolderPath,'*.mat'));
for iFile = 1:length(trackDataFiles)
    trackDataFileName = trackDataFiles(iFile).name;
    trkdataStruct = loadTrkdata(trackDataFileName, radarDataFolderPath);
    trkdata = trkdataStruct.trkdata;
    for example_number = 1:length(trkdata)
        time_stamps_array = [time_stamps_array trkdata(example_number).measurement];
    end
end


unique_timesatmps_array = unique(time_stamps_array);
unique_timesatmps_array = unique_timesatmps_array(:);
% All the examples were taken within 74 unique timestamps