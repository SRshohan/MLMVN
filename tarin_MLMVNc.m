% train_with_NetLearnC.m
disp('========== Starting MLMVN Training ==========');

% Load data
disp('Loading data files...');
load('/Users/sohanurrahman/Desktop/MLMVN/processed/LearningFourier.mat');
load('/Users/sohanurrahman/Desktop/MLMVN/processed/TestingFourier.mat');
disp(['Loaded data: ', num2str(size(LearningFourier,1)), ' training samples, ', num2str(size(TestingFourier,1)), ' testing samples']);

% Reduce feature dimensions to a manageable size
disp('Reducing feature dimensions...');
feature_count = 1000; % Adjust as needed
LearningFourier = LearningFourier(:, 1:feature_count);
TestingFourier = TestingFourier(:, 1:feature_count);
disp(['Feature dimension reduced to ', num2str(feature_count)]);

% Parameters for Net_learnC
disp('Setting network parameters...');
hidneur_num = 512; % Number of hidden neurons
outneur_num = 3;   % Number of output neurons (for 3 classes)
sec_nums = [2,2,2]; % Sectors per output neuron
RMSE_thresh = 1.04;
local_thresh = 0;
disp(['Hidden neurons: ', num2str(hidneur_num), ', Output neurons: ', num2str(outneur_num)]);

% Initial learning with first batch
disp('========== INITIAL TRAINING (Class 0) ==========');
disp(['Training on samples 1-300 (Class 0)...']);
tic;
[hidneur_weights, outneur_weights, iterations] = Net_learnC(LearningFourier(1:300,:), hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
training_time = toc;
disp(['Initial training completed in ', num2str(training_time), ' seconds']);
disp(['Required ', num2str(iterations), ' iterations']);

% Save weights
% save('trained_weights.mat', 'hidneur_weights', 'outneur_weights');

% You could also implement batch learning as in your example:
ITERATION = 0;
recognition = zeros(1,10);
flag = true;
StoppingErrorThreshold = 0.98;
disp(['Setting stopping threshold to ', num2str(StoppingErrorThreshold)]);

disp('========== STARTING BATCH TRAINING LOOP ==========');
while(flag)
    ITERATION = ITERATION + 1;
    disp(['GLOBAL ITERATION ', num2str(ITERATION), ' STARTED']);
    
    % Process remaining batches with Net_learn_testcodeC
    for x = 1:2  % For classes 2 and 3 (since class 0 was used to initialize)
        start_idx = x * 300 + 1;
        end_idx = (x+1) * 300;
        
        disp(['Processing batch for class ', num2str(x), ' (samples ', num2str(start_idx), '-', num2str(end_idx), ')...']);
        tic;
        [hidneur_weights, outneur_weights, iterations] = Net_learn_testcodeC(LearningFourier(start_idx:end_idx,:), hidneur_weights, hidneur_num, outneur_weights, outneur_num, sec_nums, RMSE_thresh, local_thresh);
        batch_time = toc;
        disp(['  Batch ', num2str(x+1), ' completed in ', num2str(batch_time), ' seconds, ', num2str(iterations), ' iterations']);
    end
    
    % Save weights
    disp('Saving current network weights...');
    save('weights_iter.mat', 'hidneur_weights', 'outneur_weights');
    
    % Test
    disp('========== TESTING CURRENT MODEL ==========');
    disp(['Testing on ', num2str(size(TestingFourier,1)), ' samples...']);
    tic;
    results = Net_testC(TestingFourier, hidneur_weights, outneur_weights, 3*pi/2);
    test_time = toc;
    disp(['Testing completed in ', num2str(test_time), ' seconds']);
    
    disp(['ITERATION ', num2str(ITERATION), ': Recognition rate = ', num2str(results*100, '%6.2f'), '%']);
    
    recognition(ITERATION) = results;
    
    % Check for stopping condition
    if results > StoppingErrorThreshold
        disp(['Recognition rate ', num2str(results*100, '%6.2f'), '% exceeds threshold ', num2str(StoppingErrorThreshold*100, '%6.2f'), '%']);
        disp('STOPPING CRITERION MET');
        flag = false;
    else
        disp(['Recognition rate ', num2str(results*100, '%6.2f'), '% below threshold ', num2str(StoppingErrorThreshold*100, '%6.2f'), '%']);
        disp('Continuing to next iteration...');
    end
    disp('----------------------------------------');
end

disp('========== TRAINING COMPLETE ==========');
disp(['Final recognition rate: ', num2str(recognition(ITERATION)*100, '%6.2f'), '%']);
disp(['Total global iterations: ', num2str(ITERATION)]);
disp('Saving final model...');
save('final_model.mat', 'hidneur_weights', 'outneur_weights', 'recognition', 'ITERATION');
disp('Done!');