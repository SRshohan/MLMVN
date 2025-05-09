% train_with_NetLearnC.m
% Load data
load('/Users/sohanurrahman/Desktop/MLMVN/processed/LearningFourier.mat');
load('/Users/sohanurrahman/Desktop/MLMVN/processed/TestingFourier.mat');

% Reduce feature dimensions to a manageable size
feature_count = 1000; % Adjust as needed
LearningFourier = LearningFourier(:, 1:feature_count);
TestingFourier = TestingFourier(:, 1:feature_count);

% Parameters for Net_learnC
hidneur_num = 512; % Number of hidden neurons
outneur_num = 3;   % Number of output neurons (for 3 classes)
sec_nums = [2,2,2]; % Sectors per output neuron
RMSE_thresh = 1.04;
local_thresh = 0;

% Initial learning with first batch
[hidneur_weights, outneur_weights, iterations] = Net_learnC(LearningFourier(1:300,:), hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);

% Save weights
% save('trained_weights.mat', 'hidneur_weights', 'outneur_weights');

% You could also implement batch learning as in your example:
ITERATION = 0;
recognition = zeros(1,10);
flag = true;
StoppingErrorThreshold = 0.98;

while(flag)
    ITERATION = ITERATION + 1;
    
    % Then process remaining batches with Net_learn_testcodeC
    for x = 1:2  % For classes 2 and 3 (since class 1 was used to initialize)
        start_idx = x * 300 + 1;
        end_idx = (x+1) * 300;
        
        [hidneur_weights, outneur_weights, iterations] = Net_learn_testcodeC(LearningFourier(start_idx:end_idx,:), hidneur_weights, hidneur_num, outneur_weights, outneur_num, sec_nums, RMSE_thresh, local_thresh);
        
        fprintf('Batch %d\n', (x+1));
    end
    
    % Save weights
    save('weights_iter.mat', 'hidneur_weights', 'outneur_weights');
    
    % Test
    results = Net_testC(TestingFourier, hidneur_weights, outneur_weights, 3*pi/2);
    
    fprintf('Iteration %d: Recognition - %6.6f\n', ITERATION, results);
    
    recognition(ITERATION) = results;
    
    if results > StoppingErrorThreshold
        flag = false;
    end
end