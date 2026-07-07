%% EXAMPLE OF A REGRESSION EXPERIMENT

% Machine Learnning Toolbox
% Author: David Nascimento Coelho
% Last mod: 2026/07/02

clear;
clc;

%% OPTIONS

% Data Options:
% SincRegression ; SinRegression ; FrankeFunction
% LinearRegression ; MultipleLinearRegression ; 
% PolynomialRegression ; 
dataName = "MultipleLinearRegression"; 	
number_of_samples = 500;
noise_std = 0.01;
randomState = 10;

% Model Options:



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
[X,Y] = mltoolbox.preprocessing.shuffle_data(X,Y);

% Split train x test
[Xtr,Xts,Ytr,Yts] = ...
    mltoolbox.preprocessing.train_test_split.split(X,Y,'train_ratio',0.7);

% Z-score Normalization
scaler = mltoolbox.preprocessing.DataScaler('mode','zscore');
Xtr_norm = scaler.fit_transform(Xtr);
Xts_norm = scaler.transform(Xts);

%% REGRESSION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.regressors.OLSRegressor();

model.fit(Xtr_norm,Ytr);

Yhat_tr = model.predict(Xtr_norm);
Yhat_ts = model.predict(Xts_norm);

%% METRICS



%% END















