%% EXAMPLE OF A CLASSIFICATION EXPERIMENT

% Machine Learnning Toolbox
% Last mod: 2026/07/02

clear;
clc;

%% OPTIONS

% Data Options
dataName = "gaussianblobs"; % linearclassification ; moons ; circles ; xor ; spirals
number_of_samples = 500;
number_of_classes = 3;
number_of_features = 2;
noise_std = 0.5;
randomState = 10;

% Pre-processing options
shuffle = true;
train_ratio = 0.7;
normalize = true;
normalization = 'zscore';

% Model options
approximation = 'theoretical';      % 'pinv' 'svd' 'theoretical'
regularization = 0.0001;            % just for theoretical aproximation

%% LOAD DATASET

data = mltoolbox.datasets.ArtificialClassificationDataset( ...
    dataName, ...
    "nSamples", number_of_samples, ...
    "nClasses", number_of_classes, ...
    "nFeatures", number_of_features, ...
    "noiseStd", noise_std, ...
    "randomState", randomState);

X  = data.inputs;
Y = data.outputs;

%% PLOT DATASET

classes = unique(Y);
colors = lines(length(classes));

figure;

hold on;
grid on;
box on;

for i = 1:length(classes)
    classIndex = Y == classes(i);

    scatter( ...
        X(classIndex, 1), ...
        X(classIndex, 2), ...
        50, ...
        colors(i, :), ...
        "filled", ...
        "MarkerFaceAlpha", 0.75, ...
        "DisplayName", sprintf("Class %d", classes(i)));
end

xlabel("Feature 1");
ylabel("Feature 2");
title("Gaussian Blobs Classification Dataset");

legend("Location", "best");
axis equal;

hold off;

%% DATA PRE-PROCESSING

% Shuffle data
if shuffle
    [X,Y] = mltoolbox.preprocessing.shuffle_data(X,Y);
end

% Split train x test
[Xtr,Xts,Ytr,Yts] = ...
    mltoolbox.preprocessing.train_test_split(X,Y,...
    'train_ratio',train_ratio, ...
    'shuffle',true);

% Normalization
if normalize
    xScaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
end

% Plot Train and Test Data


%% END