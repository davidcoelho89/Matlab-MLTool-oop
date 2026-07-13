%% EXAMPLE OF A REGRESSION EXPERIMENT

% Machine Learnning Toolbox
% Author: David Nascimento Coelho
% Last mod: 2026/07/02

clear;
clc;

%% OPTIONS

% Data Options:
dataName = "LinearRegression"; % LinearRegression MultipleLinearRegression PolynomialRegression SinRegression SincRegression FrankeFunction
number_of_samples = 100;
noise_std = 0.00;
randomState = 10;

% Pre-processing options:
shuffle = true;
train_ratio = 0.7;
normalization = 'zscore';

% Model options:
approximation = 'theoretical';     % 'pinv' 'svd' 'theoretical'
regularization = 0.0001;

%% LOAD DATASET

data = mltoolbox.datasets.ArtificialRegressionDataset(dataName, ...
                                                      "nSamples", number_of_samples, ...
                                                      "noiseStd", noise_std, ...
                                                      "randomState", randomState);
X = data.X;
Y = data.Y;

%% PLOT DATASET

%figure;
%plot(X,Y,'.');

%% DATA PRE-PROCESSING

% Shuffle data
if shuffle
    [X,Y] = mltoolbox.preprocessing.shuffle_data(X,Y);
end

% Split train x test
[Xtr,Xts,Ytr,Yts] = ...
    mltoolbox.preprocessing.train_test_split.split(X,Y,'train_ratio',train_ratio);

% Z-score Normalization
scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
Xtr_norm = scaler.fit_transform(Xtr);
Xts_norm = scaler.transform(Xts);

%% REGRESSION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.regressors.OLSRegressor('approximation',approximation, ...
                                          'regularization',regularization);

model.fit(Xtr_norm,Ytr);

Yhat_tr = model.predict(Xtr_norm);
Yhat_ts = model.predict(Xts_norm);

%% METRICS



%% END















