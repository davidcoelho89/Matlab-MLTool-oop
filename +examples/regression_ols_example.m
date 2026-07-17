%% EXAMPLE OF A REGRESSION EXPERIMENT

% Machine Learnning Toolbox
% Last mod: 2026/07/14

clear;
clc;

%% OPTIONS

% Data Options
dataName = "MultipleLinearRegressionMultiOutput"; % LinearRegression MultipleLinearRegression PolynomialRegression SinRegression SincRegression FrankeFunction MultipleLinearRegressionMultiOutput
number_of_samples = 501;
noise_std = 0.1;
randomState = 10;

% Pre-processing options
shuffle = true;
train_ratio = 0.7;
normalization = 'zscore';
normalize_input = false;
normalize_output = false;

% Model options
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
    mltoolbox.preprocessing.train_test_split.split(X,Y,...
                                                   'train_ratio',train_ratio, ...
                                                   'shuffle',true);

% figure;
% hold on
% plot(Xtr,Ytr,'r.');
% plot(Xts,Yts,'b.');
% hold off

% Normalization
if normalize_input
    xScaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
    Xtr = xScaler.fit_transform(Xtr);
    Xts = xScaler.transform(Xts);
end

if normalize_output
    yScaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
    Ytr = yScaler.fit_transform(Ytr);
    Yts = yScaler.transform(Yts);
end

if normalize_input || normalize_output
%     figure;
%     hold on
%     plot(Xtr,Ytr,'r.');
%     plot(Xts,Yts,'b.');
%     hold off
end

%% REGRESSION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.regressors.OLSRegressor('approximation',approximation, ...
                                          'regularization',regularization);

model.fit(Xtr,Ytr);

% Normalized

Yhat_tr = model.predict(Xtr);
Yhat_ts = model.predict(Xts);

figure;
plot(Ytr,Yhat_tr,'r.');
title('Ytr x Ytr-hat - Must be a line')
xlabel('Ytr')
ylabel('Ytr-hat')

figure;
plot(Yts,Yhat_ts,'b.');
title('Yts x Yts-hat - Must be a line')
xlabel('Yts')
ylabel('Yts-hat')

% Denormalized

if normalize_output
    Ytr = yScaler.inverse_transform(Ytr);
    Yhat_tr = yScaler.inverse_transform(Yhat_tr);

    figure;
    plot(Yhat_tr,Ytr,'r.');

    Yts = yScaler.inverse_transform(Yts);
    Yhat_ts = yScaler.inverse_transform(Yhat_ts);

    figure;
    plot(Yhat_ts,Yts,'b.');
end

%% METRICS

numberOfParameters = size(model.W,1);

metrics_tr = mltoolbox.metrics.regressionMetrics.calculate(Ytr,Yhat_tr,...
                               'NumParameters',numberOfParameters);
metrics_ts = mltoolbox.metrics.regressionMetrics.calculate(Yts,Yhat_ts,...
                               'NumParameters',numberOfParameters);

disp('===== REGRESSION METRICS FROM TRAINING =====');

disp('Metrics per output:');
disp(metrics_tr.perOutput);

disp('Overall metrics:');
disp(metrics_tr.overall);

disp('===== REGRESSION METRICS FROM TEST =====');

disp('Metrics per output:');
disp(metrics_ts.perOutput);

disp('Overall metrics:');
disp(metrics_ts.overall);

%% END