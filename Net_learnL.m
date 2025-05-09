

function [hidneur_weights, outneur_weights, iterations] = Net_learnL(Input, Validation, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh)
% (c) E.Aizenberg & I.Aizenberg 2014-2018
% This function runs batch learning algorithm with validation for
% MLMVN-LLS-SM n->M->k with multiple output neurons where 
% n - # of network inputs
% M - the number of neurons in a single hidden layer
% k - this network contains k output neurons with the discrete MVN
% activation function
% Hence this function is suitable for learning classification problems and
% should be used for learning k-class classification problems
%
% The learning process continues until 0 error for a validation set is
% reached
%
% The batch learning algorithm is utilized as it is presented in the paper
%
% E. Aizenberg, I. Aizenberg, �Batch LLS-based Learning Algorithm for MLMVN
% with Soft Margins�, Proceedings of the 2014 IEEE Symposium Series of
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
%           According to logic of this program sector 1 and desired output
%           1, accordingly should be reserved for recognized samples and
%           sector 0 for rejected samples (if each output neuron solves a
%           binary classification problem).
%           Network inputs must be angular values in radians. It is assumed
%           that a classification problem to be solved has continuous
%           inputs. Hence, inputs must be transformed in advance in order
%           to fit them in the interval [0, fi] where fi<2pi radians
%
% Validation - A validation set - matrix M x r of validations samples. 
%           There are M validation samples in total
%           r is the number of columns in Validation
%           Each sample consists of n = r-1 inputs followed by the desired
%           output in the r-th column. 
%           A desired output must be an integer - a class label (from 0 to
%           k-1 for k classes) 
%
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
% This fuction ether learns until either a zero error has been reached 
% along with the satisfaction of the angular soft margins criterion   
% or the user has to interrupt its execution using <Ctrl><c>
% 


%X = matrix of MVN inputs (N x n), where N=number of learning
%samples, n = number of input variables

%y_d = (N x outneur_num) matrix of desired network outputs, expressed as class labels

%hidneur_num = number of hidden neurons
%outneur_num = number of output neurons

%sec_nums = (1 x outneur_num) vector containing the number of sectors in
%each output neuron

%local_thresh = local angular threshold for determining error

win_ang = 3*pi/2;

%Use the clock to set the stream of random numbers
%RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));

% Determine the number of learning samples (N) and the number of columns in
% the matrix Input containing inputs and desired output
[N, m] = size(Input); 
%Determine the number of network inputs (input variables n)
n = m-outneur_num; % n = # of output neurons
%X = matrix of MVN inputs (N x n), where N=number of learning
%samples, n = number of input variables
X = Input(:,1:n); % X is a matrix of inputs
%y_d = (N x 1) vector of desired network outputs, expressed as class labels
y_d = Input(:,n+1:m); % y_d is a matrix of desired outputs

%Convert input values into complex numbers on the unit circle
X = exp(1i .* X);

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
phase_d = mod(phase_d, 2*pi);

%Determine sector size (angle), separately for each output neuron.
%sec_size is a (1 x outneur_num) vector
sec_size = 2*pi ./ sec_nums;

%Shift all desired phase values by half-sector counter-clockwise
for pp = 1 : outneur_num
    
    phase_d(:, pp) = phase_d(:, pp) + sec_size(pp)/2;
end


%Construct a matrix of desired network outputs (lying on the unit circle)
znet_d = exp(1j .* phase_d);


app_X = [col_app X];

% Finding of the pseudo inverse matrix of app_X
X_pinv = pinv(app_X);

iterations = 0;
nesovpad = 1;

min_nesovpad = N;

h = LearnStatsFig;
handles = guidata(h);

LearnFlag = 1;

min_RMSE = 10;


% Determine the number of validation samples (N1) and the number of columns
% in the matrix Validation containing inputs and desired outputs
[N1, m1] = size(Validation); 
%Determine the number of network inputs (input variables n)
n1 = m1-1; % n = # of inputs
%XV = matrix of MVN inputs (N1 x n), where N1=number of validation
%samples, n1 = number of input variables
XV = Validation(:,1:n1); % X is a matrix of inputs
%y_d = (N x 1) vector of desired network outputs, expressed as class labels
y_dV = Validation(:,m1); % y_d is a vector-column of desired outputs

%Convert input values into complex numbers on the unit circle
XV = exp(1i .* XV);

% Desired validation outputs in terms of phase for all output neurons

%Convert desired validation output values (y_dV), given as class labels,
%into desired phase values (phase_dV):
phase_dV = zeros(N1, outneur_num);
phase_dV = phase_dV + pi/2;

for k1 = 1:N1
    phase_dV(k1, y_dV(k1)+1) = win_ang;
end


%append a column of 1s to X from the left
%app_X
col_appV(1:N1) = 1;
col_appV = col_appV.';
app_XV = [col_appV XV];


while ( LearnFlag == 1)
    
%% Validation
%Compute the output of hidden neurons for all samples
hid_outmat = app_XV * hidneur_weights;

%Move outputs to the unit circle
hid_outmat = hid_outmat ./ abs(hid_outmat);

%append a column of 1s to hid_outmat
hid_outmat = [col_appV hid_outmat];

%Compute the output of the network
z_outneur = hid_outmat * outneur_weights;

%We will now apply the "winner take it all" principle in the following
%manner:the output neuron whose output is closest to win_ang determines the
%output class
current_phase = angle(z_outneur);

%Ensure that the angular range is [0, 2pi) instead of (-pi, pi)
current_phase = mod(current_phase, 2*pi);

win_dist = zeros(N1, outneur_num);
for ii=1:N1
    
    for pp = 1 : outneur_num
       
        win_dist(ii, pp) = abs(current_phase(ii, pp) - win_ang);

        if (win_dist(ii, pp) > pi)

            win_dist(ii, pp) = 2*pi - win_dist(ii, pp);
        end

    end
end

% calculation of the network output based on the "winner takes it all"
% technique based on the closeness of the output to the bisector of the
% desired sector
[min_dist, current_labels] = min(win_dist, [], 2);
current_labels = current_labels - 1;


Flags = (current_labels == y_dV );
sovpad = sum(Flags);

classif_rate = sovpad / N1;

%    ang_RMSE_V = 0;

    % ang_err angular errors for the entire validation set
%    ang_err = abs(current_phase - phase_dV);
    % Flags contain 1 at the positions where ang_err > pi
%    Flags = (ang_err > pi);
    % if ang_err > pi then change it to 2i - ang_err
%    ang_err(Flags) = 2*pi - ang_err(Flags);
    % ang_RMSE_V here is a sume of squared angular errors (for all output
    % neurons separately, it is a vector here)
%    ang_RMSE_V = ang_RMSE_V + sum(ang_err.^2);
    % ang_RMSE becomes actual angular RMSE averaged over all output neurons
%    ang_RMSE_V = sqrt(mean(ang_RMSE_V) / N1);  


if (classif_rate == 1) && (ang_RMSE < RMSE_thresh) 
    LearnFlag = 0;
    break
end
    
    
    %% Lrarning
    
    
    %Compute the output of hidden neurons for all samples
    hid_outmat = app_X * hidneur_weights;
    
    % Calculation the absolute values of the 
    % current hidden neurons weighted sums
    abs_hid_outmat = abs(hid_outmat);
    
    %Move outputs to the unit circle
    hid_outmat = hid_outmat ./ abs_hid_outmat;
    
    %Determine the network error
    [hid_errmat] = ErrBackProp(hid_outmat, outneur_weights, phase_d, znet_d, N, hidneur_num, outneur_num, local_thresh, abs_hid_outmat);
    
    %Adjust weights of hidden neurons
    hidneur_weights = HidNeuron_weightadj(X_pinv, hidneur_weights, hid_errmat);
    
    %Compute the output of hidden neurons for all samples
    hid_outmat = app_X * hidneur_weights;
    
    %Move outputs to the unit circle
    hid_outmat = hid_outmat ./ abs(hid_outmat);
    
     [outneur_weights, z_outneur] = OutNeuron_weightadj(hid_outmat, outneur_weights, phase_d, znet_d, N, outneur_num, local_thresh);
    
    %Compute and display learning statistics----
    iterations = iterations + 1;
    
    %error
    %err_all = sum( (znet_d - z_outneur./abs(z_outneur))' * (znet_d - z_outneur./abs(z_outneur)) );
    
    %if (err_all < min_err_all)
    %    min_err_all = err_all;
    %end
    
    
    %Determine the number of nesovpad and angular RMSE
    current_phase = angle(z_outneur);
    
    %Ensure that the angular range is [0, 2pi) instead of (-pi, pi)
    current_phase = mod(current_phase, 2*pi);
    
    % current_labels - discrete outputs (#s of sectors)
    current_labels = zeros(N, outneur_num);
    for pp = 1 : outneur_num
        
        current_labels(:, pp) = floor(current_phase(:, pp) ./ sec_size(pp));
    end

    ang_RMSE = 0;
    
    %label differences per each learning sample
    diff_labels = sum(abs(current_labels - y_d), 2);
    
    nesovpad = sum(double(diff_labels > 0));
    
    % ang_err angular errors for the entire learning set
    ang_err = abs(current_phase - phase_d);
    % Flags contain 1 at the positions where ang_err > pi
    Flags = (ang_err > pi);
    % if ang_err > pi then change it to 2i - ang_err
    ang_err(Flags) = 2*pi - ang_err(Flags);
    % ang_RMSE here is a sume of squared angular errors (for all output
    % neurons separately, it is a vector here)
    ang_RMSE = ang_RMSE + sum(ang_err.^2);
    % ang_RMSE becomes actual angular RMSE averaged over all output neurons
    ang_RMSE = sqrt(mean(ang_RMSE) / N);       
    
    if (ang_RMSE < min_RMSE)
        min_RMSE = ang_RMSE;
    end
    
    %If nesovpad == 0, stop learning
    if ( (nesovpad == 0)  && (ang_RMSE < RMSE_thresh) )
        
        LearnFlag = 0;
    end
    
    if (nesovpad < min_nesovpad)
        
        min_nesovpad = nesovpad;
    end
    
    %Display the statistic in a separate figure
    set(handles.IterLabel, 'String', num2str(iterations));
    %set(handles.ErrLabel, 'String', num2str(err_all));
    %set(handles.MinErrLabel, 'String', num2str(min_err_all));
    %set(handles.NesovpadLabel, 'String', num2str(nesovpad));
    set(handles.NesovpadLabel, 'String', num2str(sovpad));
    set(handles.MinNesovpadLabel, 'String', num2str(min_nesovpad));
 %   set(handles.MinNesovpadLabel, 'String', num2str(ang_RMSE_V));
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





