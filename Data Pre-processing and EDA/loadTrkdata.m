function trkdata = loadTrkdata(fileName, radarDataFolderPath)
    if(nargin < 2)
        radarDataFolderPath = radarDataPath;
    end
    trkdata = load(strcat(radarDataFolderPath, fileName));
end
