%% EXAMPLE OF A REGRESSION EXPERIMENT

% Machine Learnning Toolbox
% Last mod: 2026/07/14

clear;
clc;

%% OPTIONS

% Data Options:
dataName = "MultipleLinearRegression"; % LinearRegression MultipleLinearRegression PolynomialRegression SinRegression SincRegression FrankeFunction
number_of_samples = 501;
noise_std = 0.1;
randomState = 10;

% Pre-processing options:
shuffle = true;
train_ratio = 0.7;
normalization = 'zscore';

% Model options:
approximation = 'theoretical';      % 'pinv' 'svd' 'theoretical'
regularization = 0.0001;            % just for theoretical aproximation

%% LOAD DATASET

data = mltoolbox.datasets.ArtificialRegressionDataset(dataName, ...
                                                      "nSamples", number_of_samples, ...
                                                      "noiseStd", noise_std, ...
                                                      "randomState", randomState);
X = data.X;
Y = data.Y;

%% PLOT DATASET

% figure;
% plot(X,Y,'r.');

%% DATA PRE-PROCESSING

% Shuffle data
if shuffle
    [X,Y] = mltoolbox.preprocessing.shuffle_data(X,Y);
end

% Split train x test
[Xtr,Xts,Ytr,Yts] = ...
    mltoolbox.preprocessing.train_test_split.split(X,Y,'train_ratio',train_ratio);

% figure;
% hold on
% plot(Xtr,Ytr,'r.');
% plot(Xts,Yts,'b.');
% hold off

% % Normalization
% xScaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
% Xtr = xScaler.fit_transform(Xtr);
% Xts = xScaler.transform(Xts);
% yScaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
% Ytr = yScaler.fit_transform(Ytr);
% Yts = yScaler.transform(Yts);

% figure;
% hold on
% plot(Xtr,Ytr,'r.');
% plot(Xts,Yts,'b.');
% hold off

%% REGRESSION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.regressors.OLSRegressor('approximation',approximation, ...
                                          'regularization',regularization);

model.fit(Xtr,Ytr);

% Normalized

Yhat_tr = model.predict(Xtr);
Yhat_ts = model.predict(Xts);

figure;
hold on
plot(Yhat_tr,Ytr,'r.');
plot(Yhat_ts,Yts,'b.');
hold off

% Denormalized

% Ytr = yScaler.inverse_transform(Ytr);
% Yhat_tr = yScaler.inverse_transform(Yhat_tr);
% Yts = yScaler.inverse_transform(Yts);
% Yhat_ts = yScaler.inverse_transform(Yhat_ts);

figure;
hold on
plot(Yhat_tr,Ytr,'r.');
plot(Yhat_ts,Yts,'b.');
hold off

%% METRICS


                                                    
%% END