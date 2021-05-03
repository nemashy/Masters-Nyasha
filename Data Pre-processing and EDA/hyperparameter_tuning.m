hyperparameters;
radarDataPath = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';
% Filter
d = fdesign.notch('N,F0,BW,Ap',6,0,0.045,0.5);
Hd = design(d);
results = getSNRValues(fftLengths, windowLengths, overlapFractions, radarDataPath, Hd);


function results = getSNRValues(fftLengths, windowLengths, overlapFractions, radarDataPath, Hd)

    maxCombinations = length(fftLengths)*length(windowLengths)*length(overlapFractions);
    combinationNo = 1;
    trackDataFiles = dir(fullfile(radarDataPath,'*.mat'));
    results = struct('AverageSNR', cell(maxCombinations, 1), 'FFTLength', cell(maxCombinations, 1), 'WindowLength', cell(maxCombinations, 1), 'OverlapFraction', cell(maxCombinations, 1)); % Creating a table to store results
    for fftLength = windowLengths
        for windowLength = windowLengths
            if(windowLength <= fftLength)
                for overlapFraction = overlapFractions
                    sumFileAvgSNR = 0;
                    for iFile = 55:55 %length(trackDataFiles)
                        trackDataFileName = trackDataFiles(iFile).name;
                        trkdataStruct = loadTrkdata(trackDataFileName, radarDataPath);
                        trkdata = trkdataStruct.trkdata;
                        spectrograms = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength, Hd);
                        fileAvgSNR = getAvgSNR(spectrograms);
                        %sumFileAvgSNR = sumFileAvgSNR + fileAvgSNR;
                        %filesAvgSNR = sumFileAvgSNR %/iFile;
                    end
                    results(combinationNo).AverageSNR = fileAvgSNR;
                    results(combinationNo).FFTLength = fftLength;
                    results(combinationNo).WindowLength = windowLength;
                    results(combinationNo).OverlapFraction = overlapFraction;
                    disp('Finished iteration number ' + string(combinationNo))
                    combinationNo = combinationNo + 1;
                end
            end
        end
    end
end

% Finds the average SNR of spectrograms in each trkdata file
function avgSNR = getAvgSNR(spectrograms)
    sumSNR = 0;
    col = spectrograms(1).Data(:);
    noiseMeanSNR = var(col); % Looking at a noise spectrogram
    
    for posExample = 2:length(spectrograms)
        maxLeveldB = max(spectrograms(posExample).Data, [], 'all');
        sumSNR = sumSNR + maxLeveldB - noiseMeanSNR;
        avgSNR = sumSNR/posExample;
    end
    disp(avgSNR);
end
