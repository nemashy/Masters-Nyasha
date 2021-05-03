clear;

constants;
hyperparameters;
radarDataFolderPath = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';

% Filter
d = fdesign.notch('N,F0,BW,Ap',6,0,0.045,0.5);
Hd = design(d);

HAVSDatasetStruct =  struct('Data', {}, 'Label', {});    
trackDataFiles = dir(fullfile(radarDataFolderPath,'*.mat'));
for iFile = 1:length(trackDataFiles)
    trackDataFileName = trackDataFiles(iFile).name;
    trkdataStruct = loadTrkdata(trackDataFileName, radarDataFolderPath);
    trkdata = trkdataStruct.trkdata;
    spectrogramsStruct = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength, Hd);
    examplesStruct = generateExamples(spectrogramsStruct, dwellTime, exampleOverlapFraction);
    HAVSDatasetStruct = [HAVSDatasetStruct, examplesStruct];
end

save('havsdata', 'HAVSDatasetStruct', '-v7.3');

imagesc(HAVSDatasetStruct(12).Data);colorbar;

function examplesStruct = generateExamples(spectrogramsStruct, dwellTime, exampleOverlapFraction)

    examplesStruct =  struct('Data', {}, 'Label', {}); % Creating a table to store results
    
    for posSpectrogram=1:length(spectrogramsStruct)
        
        spectrogramLabel = spectrogramsStruct(posSpectrogram).Label;
        spectrogramdB = spectrogramsStruct(posSpectrogram).Data;
        spectrogramDuration = spectrogramsStruct(posSpectrogram).DurationSeconds;
        if(spectrogramDuration > dwellTime) % Check if example duration is greater than dwell time
            %spectrogramSlices = getSlices(spectrogramdB, numSlicesPerSpectrogram, dwellTime, spectrogramDuration);
            spectrogramSlices = getOverlappingSlices(spectrogramdB, exampleOverlapFraction, dwellTime, spectrogramDuration);
            % Add slices to the examples stucture
            for iSlice=1:length(spectrogramSlices)
                examplesStruct(end+1).Data = spectrogramSlices(iSlice).Data;
                examplesStruct(end).Label = spectrogramLabel;
            end
        end
    end
end

function spectrogramSlices = getSlices(spectrogramdB, numSlices, dwellTime, durationSeconds)
    [nrows, ncols] = size(spectrogramdB);
    
    numOfFramesInDwellTime = fix(dwellTime*ncols/durationSeconds);
    
    spectrogramSlices =  struct('Data', {});
    for sliceNo=1:numSlices
       startColumn = randi([1, ncols - numOfFramesInDwellTime], 1);
       stopColumn = startColumn + numOfFramesInDwellTime;
       spectrogramSlices(sliceNo).Data = spectrogramdB(:,startColumn:stopColumn);
    end

end

function spectrogramSlices = getOverlappingSlices(spectrogramdB, exampleOverlapFraction, dwellTime, durationSeconds)
    [nrows, ncols] = size(spectrogramdB);
    
    numOfFramesInDwellTime = fix(dwellTime*ncols/durationSeconds);
    
    numOfOverlappingPoints = fix(exampleOverlapFraction * numOfFramesInDwellTime);
    spectrogramSlices =  struct('Data', {});
    startCol = 1;
    for sliceNo=1:ncols
       stopCol = startCol + numOfFramesInDwellTime -1;
       if(stopCol < ncols) % Check if there are available points
           spectrogramSlices(sliceNo).Data = spectrogramdB(:,startCol:stopCol);
           startCol = stopCol - numOfOverlappingPoints + 1;
       end
    end
end
 



