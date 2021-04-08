clear

constants;
radarDataPath = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';
trackDataFiles = dir(fullfile(radarDataPath,'*.mat'));

data = {};
labels = {};

for iFile = 1:55
    disp(iFile)
    trackDataFileName = trackDataFiles(iFile).name;
    load(strcat(radarDataPath,trackDataFileName)); % Load the .mat file
    % Loading .mat file return tkrdata 
    for jRangeBin = 1:length(10)
        fsHz = trkdata(jRangeBin).PRF;
        inPhaseData = double(trkdata(jRangeBin).trk_data_real);
        quadratureData = double(1i*trkdata(jRangeBin).trk_data_imag);
        rangeBinIQData = inPhaseData + quadratureData;
        detectionClass = char(trkdata(jRangeBin).class);
    end
    
end

s = spectrogram(rangeBinIQData,hamming(128),117,128);
h = 20*log10(abs(s));
imagesc(h);
