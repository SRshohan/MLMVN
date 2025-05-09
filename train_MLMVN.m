% train_MLMVN.m
load('/Users/sohanurrahman/Desktop/MLMVN/processed/LearningFourier.mat');
load('/Users/sohanurrahman/Desktop/MLMVN/processed/TestingFourier.mat');

hidneur_num = 10;
outneur_num = 3;
sec_nums = [2 2 2];
RMSE_thresh = 0.1;
local_thresh = 0.05;

% disp(y_d(1:5,:));

size(LearningFourier)
disp(LearningFourier(1:5, 1:10)) % Show first 5 rows, first 10 columns

Input = [LearningFourier, train_labels];
Validation = [TestingFourier, test_labels];

[hidneur_weights, outneur_weights, iterations] = Net_learnL( ...
    Input, Validation, ...
    hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);

save('trained_weights.mat', 'hidneur_weights', 'outneur_weights');

size(LearningFourier)
disp(LearningFourier(1:5, 1:10)) % Show first 5 rows, first 10 columns

