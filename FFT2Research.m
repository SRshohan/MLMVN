% Directory paths taking images from
trainingPathOg = '/Users/sohanurrahman/Desktop/MLMVN/cleaned/Training';
testingPathOg  = '/Users/sohanurrahman/Desktop/MLMVN/cleaned/Testing';

% Directory paths saving new images to
% (should be a separate folder so you don’t overwrite originals)
trainingPathFin = '/Users/sohanurrahman/Desktop/MLMVN/processed/Training';
testingPathFin  = '/Users/sohanurrahman/Desktop/MLMVN/processed/Testing';

% Converting images in training folder
disp('Converting images in Training data...');
convertImg(trainingPathOg, trainingPathFin);

% Converting images in testing folder
disp('Converting images in Testing data...');
convertImg(testingPathOg,  testingPathFin);


function convertImg(inputFolder, outputFolder)
    disp(['Processing folder: ' inputFolder]);
    subfolders = {'glioma','meningioma','notumor'};
    
    for idx = 1:numel(subfolders)
        cls = subfolders{idx};
        inDir  = fullfile(inputFolder,  cls);
        outImg = fullfile(outputFolder, cls);       % for grayscale TIFFs
        outFt  = fullfile(outputFolder, [cls '_ft']);  % for .MAT FT data
        
        disp(['  Subfolder: ' cls]);
        
        % make sure output dirs exist (once per class)
        if ~exist(outImg,'dir'), mkdir(outImg); end
        if ~exist(outFt, 'dir'), mkdir(outFt);  end
        
        % look for .tif files (your pipeline writes .tif)
        files = dir(fullfile(inDir,'*.tif'));
        if isempty(files)
            disp(['    No TIFF images found in: ' inDir]);
            continue;
        end
        
        % process each image
        for i = 1:numel(files)
            imgPath = fullfile(inDir, files(i).name);
            I       = imread(imgPath);
            G       = im2gray(I);  % convert to grayscale
            
            [~, base, ~] = fileparts(files(i).name);
            
            % write out grayscale TIFF
            outTif = fullfile(outImg, [base '.tif']);
            imwrite(G, outTif, 'tiff');
            
            % compute 2D FFT and save as .MAT
            ftData = calculateFt(G);
            outMat = fullfile(outFt, [base '.mat']);
            save(outMat, 'ftData');
            
            disp(['    Converted ' files(i).name ' → ' outTif]);
            disp(['    Saved FT data → ' outMat]);
        end
    end
end

function ftData = calculateFt(image)
    % Compute 2D FFT (double precision), shift and normalize by image size
    [rows, cols] = size(image);
    F = fft2(double(image));
    F = fftshift(F);
    ftData = F / (rows * cols);
end
