
function spectrogramsStruct = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength, filterParams)
    if isa(trkdata, 'struct') % Check if the datatype is a 'struct'
            spectrogramsStruct = struct('Data',cell(length(trkdata), 1), 'Label', cell(length(trkdata), 1), 'DurationSeconds', cell(length(trkdata), 1));
            for posRangeBin = 1:length(trkdata) % posRangeBin = 16
                fsHz = trkdata(posRangeBin).PRF;
                [IQ_Sample, Sample_Label] = getIQSample(posRangeBin, filterParams, trkdata);
                % save('nyasha_signal.mat','IQ_Sample', 'Sample_Label');
                window = hamming(windowLength);
                overlapLength = overlapFraction * windowLength;
                % Get STFT of the example
                [S, F, T] = stft(IQ_Sample,fsHz,'Window',window,'OverlapLength',overlapLength,'FFTLength',fftLength);
                S_dB = 20*log10(abs(S));
                spectrogramsStruct(posRangeBin).Data = S_dB;
                spectrogramsStruct(posRangeBin).Label = Sample_Label;
                spectrogramsStruct(posRangeBin).DurationSeconds = T(end);
            end
    else
        disp('Needs a struct data type');
    end
end

function [IQ_Sample, Sample_Label] = getIQSample(rangeBinNo, filterParams, trkdata)
    inPhaseData = double(trkdata(rangeBinNo).trk_data_real);
    quadratureData = double(1i*trkdata(rangeBinNo).trk_data_imag);
    IQSample = inPhaseData + quadratureData;
    IQ_Sample = filter(filterParams,IQSample);
    Sample_Label = char(trkdata(rangeBinNo).class);
end
    