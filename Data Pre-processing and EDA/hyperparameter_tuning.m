hyperparameters;
radarDataPath = 'C:\Users\nyasha\Desktop\Thesis\Radar-Classfication-Project-master\Data\';

function SNRValues = getSNRValues(fftLengths, windowLengths, overlapFractions, radarDataPath)
    bestAvgSNR = 0;
    maxCombinations = length(fftLengths)*length(windowLengths)*length(overlapFractions);
    combinationNo = 1;
    trackDataFiles = dir(fullfile(radarDataPath,'*.mat'));
    results = struct('AverageSNR', cell(maxCombinations, 1), 'FFTLength', cell(maxCombinations, 1), 'WindowLength', cell(maxCombinations, 1), 'OverlapFraction', cell(maxCombinations, 1)); % Creating a table to store results
    for fftLength = windowLengths
        for windowLength = windowLengths
            if(windowLength <= fftLength)
                for overlapFraction = overlapFractions
                    for iFile = 1:length(trackDataFiles)
                        trackDataFileName = trackDataFiles(iFile).name;
                        trkdataStruct = loadTrkdata(trackDataFileName, radarDataPath);
                        trkdata = trkdataStruct.trkdata;
                        spectrograms = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength);
                        avgSNR = getAvgSNR(spectrograms);


                        if(avgSNR > bestAvgSNR)
                            bestResult = [avgSNR fftLength windowLength overlapFraction];
                        end
                    end

                end
                results(combinationNo).AverageSNR = bestResult(1);
                results(combinationNo).FFTLength = bestResult(2);
                results(combinationNo).WindowLength = bestResult(3);
                results(combinationNo).OverlapFraction = bestResult(4);
                
                SNRValues = results;
                disp('Finished iteration number ' + string(combinationNo))
                combinationNo = combinationNo + 1;
            end
        end
    end
end


% Finds the average SNR of spectrograms in each trkdata file
function avgSNR = getAvgSNR(spectrograms)
    sumSNR = 0;
    for posExample = 1:length(spectrograms)
        maxLeveldB = max(spectrograms(posExample).Data, [], 'all');
        minLeveldB = min(spectrograms(posExample).Data, [], 'all');
        sumSNR = sumSNR + maxLeveldB - minLeveldB;
        avgSNR = sumSNR/posExample;
    end
end
