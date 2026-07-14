%% EXAMPLE OF A SYSTEM IDENTIFICATION EXPERIMENT

% Machine Learnning Toolbox
% Last mod: 2026/07/02

clear;
clc;

%% OPTIONS

% Data Options
dataName = "ARX";

% Pre-processing options
shuffle = false;
train_ratio = 0.7;
normalization = 'zscore';

% Model options
approximation = 'theoretical';     % 'pinv' 'svd' 'theoretical'
regularization = 0.0001;

%% LOAD DATASET

data = mltoolbox.datasets.ArtificialSystemIdentificationDataset(dataName, ...
    "nSamples", 1000, ...
    "noiseStd", 0.05, ...
    "randomState", 1, ...
    "sampleTime", 0.1, ...
    "inputType", "prbs");

%% PLOT DATASET

figure

subplot(2,1,1)
plot(data.time, data.U)
ylabel("u(k)")
grid on

subplot(2,1,2)
plot(data.time, data.Y)
xlabel("Time")
ylabel("y(k)")
grid on

%% DATA PRE-PROCESSING

% Split train x test
[utr,uts,ytr,yts] = ...
    mltoolbox.preprocessing.train_test_split.split(data.U,data.Y,...
                                                   'train_ratio',train_ratio,...
                                                   'shuffle',shuffle);

% Normalization
input_scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
utr_norm = input_scaler.fit_transform(utr);
uts_norm = input_scaler.transform(uts);
output_scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
ytr_norm = output_scaler.fit_transform(ytr);
yts_norm = output_scaler.transform(yts);



%% SYSTEM IDENTIFICATION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.systemIdentification.OLSIdentifier('approximation',approximation, ...
                                                     'regularization',regularization);

model.fit(utr_norm,ytr_norm);

yhat_tr = model.predict(utr_norm,ytr_norm);
yhat_ts = model.predict(uts_norm,yts_norm);

%% METRICS



%% END

















