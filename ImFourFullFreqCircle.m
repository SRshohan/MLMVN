function [ Output ] = ImFourFullFreqCircle( Input, NFreq )
%   Extractor of Fourier coefficients corresponding to the frequencies from
%   the 1st till Nfreq-th from Fourier spectra of images stored in Input
%   Input - a matrix of vectorized images in the range [0...255](one row of
%   this matrix is a row-by-row vectorized image. In such a form images are
%   stored in the original MNIST dataset
%   NFreq - the number of frequences to be extracted from Fourier Transform
%   PhasesLearningSet - created learning set: phases exrtacted for the i-th
%   image are located in the i-th row of the matrix PhasesLearningSet

% sizes of the input matrix, n samples by M vectorized pixels
[N, M] = size(Input);
% size of a square vectorized image
n = sqrt(M);

    % s is a coordinate of the 0-th frequency specral coefficient 
    s = n/2 + 1;
    kk=0;

        % Nested loop over a "Fourier spectrum" to count the output size
        for k1 = 1:N
            for k2=1:M
                % if this condition holds, then (k1, k2) are coordinates of
                % the spectral coefficient, which shall be included.
                % If it does not hold, then just continue to the next k2
                if (sqrt((k1-s)^2 + (k2-s)^2) <= NFreq) && (k1~=s) && (k2~=s) 
                    % Increment the counter for extracted phases
                    kk = kk + 1;
                end
            end
        end


Output = zeros(N, kk);
% Creation of the container to store Fourier coefficients corresponding 
% to a single image % (will be used as a buffer)
OutputVector = zeros(1, kk);

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
    % Extraction of phase from Fourier spectrum with change of its range to
    % [0, 2pi]
    %pAF = mod(angle(AF), 2*pi);

    % Initialization of the counter for extracted phases
    kk = 0;
    
  
        % Nested loop over a Fourier spectrum
        for k1 = 1:N
            for k2=1:M
                % if this condition holds, then (k1, k2) are coordinates of
                % the spectral coefficient, which shall be included.
                % If it does not hold, then just continue to the next k2
                if (sqrt((k1-s)^2 + (k2-s)^2) <= NFreq) && (k1~=s) && (k2~=s) 
                    % Increment the counter for extracted phases
                    kk = kk + 1;
                    % Extract a spectral coefficient with
                    % the coordinates (k1, k2) and put to the buffer
                    OutputVector(kk) = AF(k1, k2);
                end
            end
        end
        
    
    % Assign the k-th row of the resulting matrix the contents of the
    % buffer (it contains phases from all frequences from the 1st to the
    % NFreq-th
    Output(k,:) = OutputVector;
    
    
end


end

