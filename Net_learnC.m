

function [hidneur_weights, outneur_weights, iterations] = Net_learnC(input, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh)

% (c) E.Aizenberg & I.Aizenberg 2014-2018
% This function runs batch learning algorithm for MLMVN-LLS-SM n->M->k with
% multiple output neurons where 
% n - # of network inputs
% M - the number of neurons in a single hidden layer
% k - this network contains k output neurons with the discrete MVN
% activation function
% Hence this function is suitable for learning classification problems and
% should be used for learning k-class classification problems
%
% The batch learning algorithm is utilized as it is presented in the paper
%
% E. Aizenberg, I. Aizenberg, “Batch LLS-based Learning Algorithm for MLMVN
% with Soft Margins”, Proceedings of the 2014 IEEE Symposium Series of
% Computational Intelligence (SSCI-2014), December, 2014, pp. 48-55.   
%
% Calling Parameters:
% Input -   matrix N x m of learning samples. 
%           There are N learning samples in total
%           m is the number of columns in Input
%           Each sample consists of n = m-outneur_num inputs followed by
%           the outneur_num desired outputs for all output neurons in the
%           columns from n+1 to m  
%           Each desired output must be an integer. Since this network
%           solves classification problems, then each output neuron has to 
%           recognize samples belonging to some certain class and reject
%           samples belonging to other classes.
%           According to logic of this program sector 0 and desired output
%           0, accordingly should be reserved for recognized samples and
%           sector 1 for rejected samples (is each output neuron solves a
%           binary classification problem).
%           Network inputs must be angular values in radians. It is assumed
%           that a classification problem to be solved has continuous
%           inputs. Hence, inputs must be transformed in advance in order
%           to fit them in the interval [0, fi] where fi<2pi radians
% hidneur_num - # of hidden neurons in a single hidden layer
% outneur_num - # of output neurons. It is assumed that if there are k
%               output neurons. For example, each of tem may solve a binary
%               classification problem 1 vs.all, which means that a k-class
%               classification problem can be considered in this way.     
%               Output neurons have a discrete activation function.
% sec_nums -    a vector containing # of sectors for output neurons. If
%               there are k output neurons, this vecor must contain k
%               components. For example, for 3 output neurons, all solving
%               a binary classification problem, this vector shall be 
%               [2, 2, 2]
% RMSE_thresh - global angular threshold in radians for angular RMSE for
%               MLMVN learning with soft margins. It must be taken less
%               than a half of the sector size (2pi/sec_nums)
% local_thresh- local threshold angular threshold in radians for
%               determining whether to count the error for a certain
%               sample. It must be taken <= RMSE_thresh. Whenever a local
%               angular error for a sample is <= local_thresh, then the
%               error for this sample is considered = 0
%
% Returning Parameters:
% hidneur_weights -     Matrix of weights of all hidden neurons
% outneur_weights -     Matrix of weights of the output neuron
% iterations -          The resulting # of learning iterations
%

%Use the clock to set the stream of random numbers
%RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

[n m]=size(input);

X = input(:, 1:m - outneur_num);
y_d = input(:, m - outneur_num + 1:end);
%Convert input values into complex numbers on the unit circle
%X = exp(1i .* X);

%Determine the number of learning samples
N = size(X, 1);

%Determine the number of input variables n
n = size(X, 2);

%Generate random weights for the hidden neurons:
hidneur_weights = zeros(n+1, hidneur_num);

for hh = 1 : hidneur_num

    %real part (interval -0.5 to 0.5)
    w_re = rand(n+1, 1) - 0.5;
    %imaginary part (interval -0.5 to 0.5)
    w_im = rand(n+1, 1) - 0.5;

    %Construct a weights vector, dimensions (n+1 x 1)
    hidneur_weights(:, hh) = w_re + 1i .* w_im;

end


%Generate random weights for the output neurons:
outneur_weights = zeros(hidneur_num+1, outneur_num);

for pp = 1 : outneur_num
    
    %real part (interval -0.5 to 0.5)
    w_re = rand(hidneur_num+1, 1) - 0.5;
    %imaginary part (interval -0.5 to 0.5)
    w_im = rand(hidneur_num+1, 1) - 0.5;
    
    %Construct a weights vector, dimensions (hidneur_num+1 x 1)
    outneur_weights(:, pp) = w_re + 1i .* w_im;
end


%----

%Convert desired network output values (y_d), given as class labels, into
%desired phase values (phase_d):
phase_d = zeros(N, outneur_num);
for pp = 1 : outneur_num
    
    phase_d(:, pp) = y_d(:, pp) .* (2*pi/sec_nums(pp));
end

%Ensure that the angular range is [0, 2pi) instead of (-pi, pi)
for ii=1:N
    for pp = 1 : outneur_num
        
        if (phase_d(ii, pp) < 0)
            phase_d(ii, pp) = phase_d(ii, pp) + 2*pi;
        end
    end
end

%Determine sector size (angle), separately for each output neuron.
%sec_size is a (1 x outneur_num) vector
sec_size = 2*pi ./ sec_nums;

%Shift all desired phase values by half-sector counter-clockwise
for pp = 1 : outneur_num
    
    phase_d(:, pp) = phase_d(:, pp) + sec_size(pp)/2;
end


%Construct a matrix of desired network outputs (lying on the unit circle)
znet_d = exp(1j .* phase_d);


%append a column of 1s to X from the left, yielding a (N x n+1) matrix
%app_X
col_app(1:N) = 1;
col_app = col_app.';
app_X = [col_app X];

%Pre-compute the SVD of app_X and the pseudo-inverse of app_X. The latter
%will be used during LLS adjustment of hidden neuron weights.
%Compute the full SVD of X
%[U,S,V] = svd(app_X);

%Let M = n+1
%M = n+1;

%Retain only the first M columns of U, and first M rows of S
%U_hat = U(:, 1:M);
%S_hat = S(1:M, :); %S_hat becomes an M x M square matrix

%Construct the pseudo-inverse of S
%S_hpinv = diag(1 ./ diag(S_hat));

%Construct the pseudo-inverse of X
%X_pinv = V * S_hpinv * U_hat';

X_pinv = pinv(app_X);

iterations = 0;
nesovpad = 1;

min_nesovpad = N;

h = LearnStatsFig;
handles = guidata(h);

LearnFlag = 1;

min_err_all = flintmax;
min_RMSE = flintmax;

N_x_outneur_num = N * outneur_num;

while ( LearnFlag == 1)
    
    
    %Compute the output of hidden neurons for all samples
    hid_outmat = app_X * hidneur_weights;
    
    % Calculation the absolute values of the 
    % current hidden neurons weighted sums
    abs_hid_outmat = abs(hid_outmat);
    
    %Move outputs to the unit circle
    hid_outmat = hid_outmat ./ abs_hid_outmat;
    
    %Determine the network error
    [hid_errmat] = ErrBackProp(hid_outmat, outneur_weights, phase_d, znet_d, y_d, sec_size, N, hidneur_num, outneur_num, local_thresh, abs_hid_outmat);
    
    %Adjust weights of hidden neurons
    for hh = 1 : hidneur_num
        
        %hidneur_weights(:, hh) = HidNeuron_weightadj(X, hidneur_weights(:, hh), hid_errmat(:, hh), N);
        hidneur_weights(:, hh) = HidNeuron_weightadj(X, X_pinv, hidneur_weights(:, hh), hid_errmat(:, hh), N);
        
    end
   
    %Compute the output of hidden neurons for all samples
    hid_outmat = app_X * hidneur_weights;
    
    %Move outputs to the unit circle
    hid_outmat = hid_outmat ./ abs(hid_outmat);
    
    [outneur_weights, z_outneur] = OutNeuron_weightadj(hid_outmat, outneur_weights, phase_d, znet_d, y_d, sec_size, N, outneur_num, local_thresh);
    
    %Compute and display learning statistics----
    iterations = iterations + 1;
    
    %error
    %err_all = sum( (znet_d - z_outneur./abs(z_outneur))' * (znet_d - z_outneur./abs(z_outneur)) );
    
    %if (err_all < min_err_all)
    %    min_err_all = err_all;
    %end
    
    
    %Determine the number of nesovpad and angular RMSE
    current_phase = angle(z_outneur);
    
    for ii=1:N

        for pp = 1 : outneur_num

            if (current_phase(ii, pp) < 0)
                current_phase(ii, pp) = current_phase(ii, pp) + 2*pi;
            end
        end
    end

    current_labels = zeros(N, outneur_num);
    for pp = 1 : outneur_num
        
        current_labels(:, pp) = floor(current_phase(:, pp) ./ sec_size(pp));
    end

    ang_RMSE = 0;
    
    %label differences per each learning sample
    diff_labels = sum(abs(current_labels - y_d), 2);
    
    nesovpad = sum(double(diff_labels > 0));
    
    for ii=1:N
    
        for pp = 1 : outneur_num
            
            ang_err = abs(current_phase(ii, pp) - phase_d(ii, pp));

            if (ang_err > pi)

                ang_err = 2*pi - ang_err;
            end

            ang_RMSE = ang_RMSE + ang_err^2;
        end
        
        %if (diff_labels(ii) > 0)
        %    nesovpad = nesovpad + 1;
        %end
    end
    
    
    ang_RMSE = sqrt(ang_RMSE / N_x_outneur_num);
    
    
    if (ang_RMSE < min_RMSE)
        min_RMSE = ang_RMSE;
    end
    
    %If nesovpad == 0, stop learning
    if ( (nesovpad ==0 )  && (ang_RMSE < RMSE_thresh) )
        
        LearnFlag = 0;
    end
    
    if (nesovpad < min_nesovpad)
        
        min_nesovpad = nesovpad;
    end
    
    %Display the statistic in a separate figure
    set(handles.IterLabel, 'String', num2str(iterations));
    %set(handles.ErrLabel, 'String', num2str(err_all));
    %set(handles.MinErrLabel, 'String', num2str(min_err_all));
    set(handles.NesovpadLabel, 'String', num2str(nesovpad));
    set(handles.MinNesovpadLabel, 'String', num2str(min_nesovpad));
    set(handles.AngRMSELabel, 'String', num2str(ang_RMSE));
    set(handles.MinAngRMSELabel, 'String', num2str(min_RMSE));
    guidata(h, handles);
    drawnow;
    
    
    %Build a list of statistic (for all iterations)
    %verr_all(iterations) = err_all;
    %vang_RMSE(iterations) = ang_RMSE;
    %vnesovpad(iterations) = nesovpad;
    %----
end

close(h);

disp(' ');
disp(['Iteration: ', num2str(iterations)]);
%disp(['Squared norm of error ', num2str(err_all)]);
disp(['Nesovapd: ', num2str(nesovpad)]);
disp(['Ang RMSE: ', num2str(ang_RMSE)]);

disp('Learning completed!');





