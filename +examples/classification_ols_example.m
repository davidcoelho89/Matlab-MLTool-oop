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
    Xtr = xScaler.fit_transform(Xtr);
    Xts = xScaler.transform(Xts);
end

%% PLOT TRAIN AND TEST DATA

if number_of_features ~= 2
    warning(['The graphical representation uses only the first two ' ...
        'features of the dataset.']);
end

% Use all classes so that train and test have the same colors
classes = unique([Ytr; Yts]);
colors = lines(length(classes));

figure;

t = tiledlayout(1, 2, ...
    "TileSpacing", "compact", ...
    "Padding", "compact");

% ---------------------------------------------------------
% Training data
% ---------------------------------------------------------
axTrain = nexttile;
hold(axTrain, "on");
grid(axTrain, "on");
box(axTrain, "on");

for i = 1:length(classes)
    classIndex = Ytr == classes(i);

    scatter(axTrain, ...
        Xtr(classIndex, 1), ...
        Xtr(classIndex, 2), ...
        50, ...
        colors(i, :), ...
        "filled", ...
        "MarkerFaceAlpha", 0.75, ...
        "MarkerEdgeColor", "k", ...
        "LineWidth", 0.4, ...
        "DisplayName", sprintf("Class %d", classes(i)));
end

xlabel(axTrain, "Feature 1");
ylabel(axTrain, "Feature 2");
title(axTrain, sprintf( ...
    "Training data — %d samples", size(Xtr, 1)));

legend(axTrain, ...
    "Location", "best", ...
    "NumColumns", 1);

axis(axTrain, "equal");
hold(axTrain, "off");

% ---------------------------------------------------------
% Test data
% ---------------------------------------------------------
axTest = nexttile;
hold(axTest, "on");
grid(axTest, "on");
box(axTest, "on");

for i = 1:length(classes)
    classIndex = Yts == classes(i);

    scatter(axTest, ...
        Xts(classIndex, 1), ...
        Xts(classIndex, 2), ...
        50, ...
        colors(i, :), ...
        "filled", ...
        "MarkerFaceAlpha", 0.75, ...
        "MarkerEdgeColor", "k", ...
        "LineWidth", 0.4, ...
        "DisplayName", sprintf("Class %d", classes(i)));
end

xlabel(axTest, "Feature 1");
ylabel(axTest, "Feature 2");
title(axTest, sprintf( ...
    "Test data — %d samples", size(Xts, 1)));

legend(axTest, ...
    "Location", "best", ...
    "NumColumns", 1);

axis(axTest, "equal");
hold(axTest, "off");

% Use identical axis limits in both plots
linkaxes([axTrain, axTest], "xy");

title(t, sprintf( ...
    "%s classification dataset — %.0f%% train / %.0f%% test", ...
    dataName, ...
    100 * train_ratio, ...
    100 * (1 - train_ratio)));

%% CLASSIFICATION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.classifiers.OLSClassifier('approximation',approximation, ...
                                            'regularization',regularization);

model.fit(Xtr,Ytr);

Yhat_tr = model.predict(Xtr);
Yhat_ts = model.predict(Xts);

%% METRICS

classLabels = model.classLabels(:)';

Ytr_onehot    = double(Ytr(:)    == classLabels);
Yhat_tr_onehot = double(Yhat_tr(:) == classLabels);

Yts_onehot    = double(Yts(:)    == classLabels);
Yhat_ts_onehot = double(Yhat_ts(:) == classLabels);

metrics_tr = mltoolbox.metrics.classificationMetrics.calculate( ...
    Ytr_onehot, Yhat_tr_onehot, ...
    'ClassLabels', classLabels);

metrics_ts = mltoolbox.metrics.classificationMetrics.calculate( ...
    Yts_onehot, Yhat_ts_onehot, ...
    'ClassLabels', classLabels);

disp('===== CLASSIFICATION METRICS: TRAINING =====');
disp(metrics_tr);

disp('===== CLASSIFICATION METRICS: TEST =====');
disp(metrics_ts);

%% END