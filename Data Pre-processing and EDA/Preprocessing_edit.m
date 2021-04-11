clear

constants;
hyperparameters;
radarDataFolderPath = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';

HAVSDatasetStruct =  struct('Data', {}, 'Label', {});    
trackDataFiles = dir(fullfile(radarDataFolderPath,'*.mat'));
for iFile = 1:length(trackDataFiles)
    trackDataFileName = trackDataFiles(iFile).name;
    trkdataStruct = loadTrkdata(trackDataFileName, radarDataFolderPath);
    trkdata = trkdataStruct.trkdata;
    spectrogramsStruct = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength);
    examplesStruct = generateExamples(spectrogramsStruct, dwellTime, numSlicesPerSpectrogram);
    HAVSDatasetStruct = [HAVSDatasetStruct, examplesStruct];
end
save('havsdata', 'HAVSDatasetStruct', '-v7.3');

function examplesStruct = generateExamples(spectrogramsStruct, dwellTime, numSlicesPerSpectrogram)

    examplesStruct =  struct('Data', {}, 'Label', {}); % Creating a table to store results
    
    for posSpectrogram=1:length(spectrogramsStruct)
        
        spectrogramLabel = spectrogramsStruct(posSpectrogram).Label;
        spectrogramdB = spectrogramsStruct(posSpectrogram).Data;
        spectrogramDuration = spectrogramsStruct(posSpectrogram).DurationSeconds;
        if(spectrogramDuration > dwellTime) % Check if example duration is greater than dwell time
            spectrogramSlices = getSlices(spectrogramdB, numSlicesPerSpectrogram, dwellTime, spectrogramDuration);
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

function numOfFrames = getNumOfFrames(dwellTime, Fs)
    numOfFrames = dwellTime*Fs;
end


% Loads the trkdata file
function trkdata = loadTrkdata(fileName, radarDataFolderPath)
    if(nargin < 2)
        radarDataFolderPath = radarDataPath;
    end
    trkdata = load(strcat(radarDataFolderPath, fileName));
end

% Converts raw IQ samples to spectrograms
function spectrogramsStruct = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength)
    if isa(trkdata, 'struct') % Check if the datatype is a 'struct'
            spectrogramsStruct = struct('Data',cell(length(trkdata), 1), 'Label', cell(length(trkdata), 1), 'DurationSeconds', cell(length(trkdata), 1));
            for posRangeBin = 1:length(trkdata)
                fsHz = trkdata(posRangeBin).PRF;
                inPhaseData = double(trkdata(posRangeBin).trk_data_real);
                quadratureData = double(1i*trkdata(posRangeBin).trk_data_imag);
                IQSample = inPhaseData + quadratureData;
                sampleLabel = char(trkdata(posRangeBin).class);

                window = hamming(windowLength);
                overlapLength = overlapFraction * windowLength;
                % Get STFT of the example
                [S, F, T] = stft(IQSample,fsHz,'Window',window,'OverlapLength',overlapLength,'FFTLength',fftLength);
                S_dB = 20*log10(abs(S));
                spectrogramsStruct(posRangeBin).Data = S_dB;
                spectrogramsStruct(posRangeBin).Label = sampleLabel;
                spectrogramsStruct(posRangeBin).DurationSeconds = max(T);
            end
    else
        disp('Needs a struct data type');
    end
end


