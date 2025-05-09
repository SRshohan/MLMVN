% train_MLMVN.m
load('/Users/sohanurrahman/Desktop/MLMVN/processed/LearningFourier.mat');
load('/Users/sohanurrahman/Desktop/MLMVN/processed/TestingFourier.mat');

% Reduce feature dimensions
feature_count = 100; % Adjust as needed
LearningFourier = LearningFourier(:, 1:feature_count);
TestingFourier = TestingFourier(:, 1:feature_count);

% Create label vectors
train_labels = [zeros(300,1); ones(300,1); 2*ones(300,1)];
test_labels = [zeros(134,1); ones(134,1); 2*ones(133,1)];

% Parameters
hidneur_num = 100; % Increase number of hidden neurons
outneur_num = 1;
sec_nums = [2, 2, 2];
RMSE_thresh = 0.1;
local_thresh = 0.05;

% Initialize weights randomly (will be updated in batches)
hidneur_weights = [];
outneur_weights = [];

% Batch parameters
batch_size = 20;
num_batches = ceil(size(LearningFourier, 1) / batch_size);

% Training loop with batches
for epoch = 1:5 % Number of epochs
    fprintf('Epoch %d\n', epoch);
    
    for b = 1:num_batches
        % Get batch indices
        start_idx = (b-1) * batch_size + 1;
        end_idx = min(b * batch_size, size(LearningFourier, 1));
        
        % Extract batch data
        batch_features = LearningFourier(start_idx:end_idx, :);
        batch_labels = train_labels(start_idx:end_idx);
        
        % Create input for this batch
        Input_batch = [batch_features, batch_labels];
        Input_batch(:,end) = real(Input_batch(:,end));
        
        % Use a subset of test data for validation
        test_subset_size = min(100, size(TestingFourier, 1));
        Validation_batch = [TestingFourier(1:test_subset_size,:), test_labels(1:test_subset_size)];
        Validation_batch(:,end) = real(Validation_batch(:,end));
        
        fprintf('  Batch %d/%d (samples %d-%d)\n', b, num_batches, start_idx, end_idx);
        
        % Train on this batch (passing current weights if available)
        if isempty(hidneur_weights) || isempty(outneur_weights)
            [hidneur_weights, outneur_weights, iterations] = Net_learnL(Input_batch, Validation_batch, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
        else
            % You'd need a version of Net_learnL that accepts initial weights
            % This might require a small modification to Net_learnL
            % For now, we'll just call the standard version
            [hidneur_weights, outneur_weights, iterations] = Net_learnL(Input_batch, Validation_batch, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
        end
    end
end

% Final evaluation on full test set
% (This would require a separate evaluation function)

% Save the final weights
save('trained_weights.mat', 'hidneur_weights', 'outneur_weights');