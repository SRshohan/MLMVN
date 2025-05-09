function [ Output ] = ImFourFullFreqZigzag(Input, NFreq )
%   Extractor of Fourier coefficients corresponding to the frequencies from
%   the 1st till Nfreq-th from Fourier spectra of images stored in Input
%   Input - a matrix of vectorized images in the range [0...255](one row of
%   this matrix is a row-by-row vectorized image. In such a form images are
%   stored in the original MNIST dataset
%   NFreq - the number of frequences to be extracted from Fourier Transform
%   Output - created learning set: Fourier coefficients exrtacted for the 
%   i-th image are located in the i-th row of Output

% sizes of the input matrix, n samples by M vectorized pixels
[N, M] = size(Input);
% size of a square vectorized image
n = sqrt(M);

% Calculation of the number of Fourier coefficients to be extracted as the 
% number of % spectral coefficients in 2D frequencies from the 1st till the
% NFreq-th one as the sum of the first NFreq members of the arithmetic 
% progression % with the 1st member 4 (the number of coefficients at the 
% 1st frequency) and the difference 4
NExtractedCoefficients = ((2*4 + 4*(NFreq-1))/2)* NFreq;
% Creation of the resulting matrix containing the same number of rows as
% Input and the number of columns equal to the number of phases
% corresponding to the drequencies from 1st till NFreq-th 
Output = zeros(N, NExtractedCoefficients);
% Creation of the container for phases corresponding to a single image
% (will be used as a buffer)
OutputVector = zeros(1, NExtractedCoefficients);

% A loop over rows of Input
for k = 1 : N
    % A is assigned the k-th vectorized image (here it is a vector)
    A = Input(k, 1:M);
    % A is reshaped to a 2D n x n image (here it becomes an n x n matrix) 
    A = reshape(A, [n, n]);
    % 2D Fourier transform of A
    AF = fft2(A);
    AF = AF / (n^2);
    % Circular shift of the 2D Fouriier transform to put 0-th frequency in 
    % the middle  
    AF = fftshift(AF);
    % s is a coordinate of the 0-th frequency specral coefficient 
    s = n/2 + 1;
    % Initialization of the counter for extracted Fourier coefficients
    kk = 0;
    
    % A for loop over all frequencies to be extracted
    % Extraction of each frequency coefficients is performed within a 
    % (s-freq) x (s+freq) window
    for freq = 1:NFreq
        
        s1 = s-freq;
        s2 = s+freq;
        % Nested loop over a(s-freq) x (s+freq) window
        for k1 = s1:s2
            for k2=s1:s2
                % if this condition holds, then (k1, k2) are coordinates of
                % the spectral coefficient belonging to the freq frequency.
                % If it does not hold, then just continue to the next k2
                if ((abs(s-k1) + abs(s-k2)) == freq)
                    % Increment the counter for extracted phases
                    kk = kk + 1;
                    % Extract a spectral coefficient with
                    % the coordinates (k1, k2) and put to the buffer
                    OutputVector(kk) = AF(k1, k2);
                end
            end
        end
        
    end
    % Assign the k-th row of the resulting matrix the contents of the
    % buffer (it contains phases from all frequences from the 1st to the
    % NFreq-th
    Output(k,:) = OutputVector;
    
    
end


end

