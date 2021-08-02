% Load parameters
params;

results = getSNRValues(fftLengths, windowLengths, overlapFractions, data_dir, filter_params);


function results = getSNRValues(fftLengths, windowLengths, overlapFractions, data_dir, filter_params)

    maxCombinations = length(fftLengths)*length(windowLengths)*length(overlapFractions);
    combinationNo = 1;
    trackDataFiles = dir(fullfile(data_dir,'*.mat'));
    results = struct('PSNR', cell(maxCombinations, 1), 'FFTLength', cell(maxCombinations, 1), 'WindowLength', cell(maxCombinations, 1), 'OverlapFraction', cell(maxCombinations, 1)); % Creating a table to store results
    for fftLength = windowLengths
        for windowLength = windowLengths
            if(windowLength <= fftLength)
                for overlapFraction = overlapFractions
                    for iFile = 55:55 %length(trackDataFiles)
                        trackDataFileName = trackDataFiles(iFile).name;
                        trkdataStruct = loadTrkdata(trackDataFileName, data_dir);
                        trkdata = trkdataStruct.trkdata;
                        spectrograms = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength, filter_params);
                        PSNR = getPSNR(spectrograms);
                    end
                    results(combinationNo).PSNR = PSNR;
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
function PSNR = getPSNR(spectrograms)
    maxLevel_dB = 0;
    for posExample = 1:length(spectrograms)
        if(strcmp(spectrograms(posExample).Label, "clutter/noise"))
            Noise_dB = 10*log10(var(spectrograms(posExample).Data(:)));
        else
            Current_Max_Level_dB = max(20*log10(abs(spectrograms(posExample).Data)), [], 'all');
            if(Current_Max_Level_dB > maxLevel_dB)
                maxLevel_dB = Current_Max_Level_dB;           
            end
        end    
    end
    PSNR = maxLevel_dB - Noise_dB;
    disp(Noise_dB);
    disp(length(spectrograms));
end
