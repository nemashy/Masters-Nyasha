function [trk_data] = load_trk_data(data_dir,file_name)
% This function loads the raw data stored in .mat files
% Each .mat file corresponds to a certain time
    file_path = strcat(data_dir,'\',file_name);
    trk_data_struct = load(file_path);
    trk_data = trk_data_struct.trkdata;
end

