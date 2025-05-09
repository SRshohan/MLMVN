% -------------------------------------------------------------------------
% Main script: compute average tumor sizes and collect FT data
% -------------------------------------------------------------------------

% Declare globals so both the script and the function refer to the same arrays
global LearningFourier TestingFourier;
LearningFourier = [];
TestingFourier  = [];

% Root folder containing cleaned/Training and cleaned/Testing
inputFolder = '/Users/sohanurrahman/Desktop/MLMVN/processed';

% Subfolder names
subfolders1 = {'glioma_ft','meningioma_ft','notumor_ft'};
subfolders2 = {'Training','Testing'};

% How many samples per class
sampleSize = 300;

% Tumor dimensions [width, height] in pixels
tumorDimensions = [
    93 86;    % glioma
    78 58;    % meningioma
    43 35     % notumor
];

% Loop over each tumor type
for t = 1:numel(subfolders1)
    tumor = subfolders1{t};
    dims  = tumorDimensions(t,:);
    
    % Calculate average tumor size (in pixels) from your Fourier patches
    avgSize = calculateAverageTumorSize( ...
        inputFolder, tumor, subfolders2, sampleSize, dims);
    
    % Compute a cutoff frequency as 1/sqrt(area)
    cutoff = 1 / sqrt(avgSize);
    
    fprintf('Tumor type: %s\n', tumor);
    fprintf('  Avg Tumor Size (pixels): %.2f\n', avgSize);
    fprintf('  Cutoff Frequency: %.4f\n\n', cutoff);
end

% Save out the accumulated Fourier data
save(fullfile(inputFolder,'LearningFourier.mat'),'LearningFourier');
save(fullfile(inputFolder,'TestingFourier.mat'), 'TestingFourier');



% -------------------------------------------------------------------------
% calculateAverageTumorSize
%   Walks through Training & Testing .mat files for one tumor type,
%   crops a centered h×w patch (dims), thresholds it, and returns mean area.
%   Also appends each full ftData(:)' into the global arrays.
% -------------------------------------------------------------------------
function avgSize = calculateAverageTumorSize(rootFolder, tumorType, categories, sampleSize, dims)
    global LearningFourier TestingFourier

    tumorSizes = [];
    
    for c = 1:numel(categories)
        cat    = categories{c};                           % 'Training' or 'Testing'
        folder = fullfile(rootFolder, cat, tumorType);
        mats   = dir(fullfile(folder,'*.mat'));
        
        if isempty(mats)
            fprintf('No MAT files found in: %s\n', folder);
            continue;
        end
        
        nFiles = min(sampleSize, numel(mats));
        for i = 1:nFiles
            data   = load(fullfile(folder, mats(i).name), 'ftData');
            ft     = data.ftData;                         % complex 2D spectrum
            [rows, cols] = size(ft);
            
            % Crop a centered rectangle of size [width, height] = dims
            w = dims(1);  h = dims(2);
            cr = round(rows/2);  cc = round(cols/2);
            r1 = max(1, cr - round(h/2));  r2 = min(rows, cr + round(h/2));
            c1 = max(1, cc - round(w/2));  c2 = min(cols, cc + round(w/2));
            patch = ft(r1:r2, c1:c2);
            
            % Count "on" pixels above a small magnitude threshold
            tumorSizes(end+1,1) = findTumorSize(patch);
            
            % Flatten the full-spectrum and append to the right global
            rowVec = ft(:).';
            if strcmp(cat, 'Training')
                LearningFourier(end+1,:) = rowVec;
            else
                TestingFourier(end+1,:) = rowVec;
            end
        end
    end
    
    % Return the mean tumor area
    avgSize = mean(tumorSizes);
end


% -------------------------------------------------------------------------
% findTumorSize
%   Simple threshold+count on the magnitude of a cropped FT patch
% -------------------------------------------------------------------------
function sz = findTumorSize(patch)
    thresh = 1e-7;
    mask   = abs(patch) > thresh;
    sz     = sum(mask(:));
end

