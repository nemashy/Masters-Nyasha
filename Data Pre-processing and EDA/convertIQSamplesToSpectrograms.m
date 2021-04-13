
function spectrogramsStruct = convertIQSamplesToSpectrograms(trkdata, windowLength, overlapFraction, fftLength, Hd)
    if isa(trkdata, 'struct') % Check if the datatype is a 'struct'
            spectrogramsStruct = struct('Data',cell(length(trkdata), 1), 'Label', cell(length(trkdata), 1), 'DurationSeconds', cell(length(trkdata), 1));
            for posRangeBin = 1:length(trkdata)
                fsHz = trkdata(posRangeBin).PRF;
                inPhaseData = double(trkdata(posRangeBin).trk_data_real);
                quadratureData = double(1i*trkdata(posRangeBin).trk_data_imag);
                IQSample = inPhaseData + quadratureData;
                IQSample = filter(Hd,IQSample);
                sampleLabel = char(trkdata(posRangeBin).class);

                window = hamming(windowLength);
                overlapLength = overlapFraction * windowLength;
                % Get STFT of the example
                [S, F, T] = stft(IQSample,fsHz,'Window',window,'OverlapLength',overlapLength,'FFTLength',fftLength);
                S_dB = 20*log10(abs(S));
                spectrogramsStruct(posRangeBin).Data = S_dB;
                spectrogramsStruct(posRangeBin).Label = sampleLabel;
                spectrogramsStruct(posRangeBin).DurationSeconds = T(end);
            end
    else
        disp('Needs a struct data type');
    end
end