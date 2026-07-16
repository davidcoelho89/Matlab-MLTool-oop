%% EXAMPLE OF A SYSTEM IDENTIFICATION EXPERIMENT

% Machine Learnning Toolbox
% Last mod: 2026/07/14

clear;
clc;

%% OPTIONS

% Data Options
dataName = "ARX";
number_of_samples = 500;
noise_std = 0.05;
randomState = 1;
sampleTime = 0.1;
inputType = "prbs";	% prbs whitenoise step sine chirp

% Pre-processing options
shuffle = false;
train_ratio = 0.7;
normalization = 'zscore';
normalize_input = false;
normalize_output = false;

% Model options
approximation = 'theoretical';	% 'pinv' 'svd' 'theoretical'
regularization = 0.0001;

%% LOAD DATASET

data = mltoolbox.datasets.ArtificialSystemIdentificationDataset(dataName, ...
       "nSamples", number_of_samples, ...
       "noiseStd", noise_std, ...
       "randomState", randomState, ...
       "sampleTime", sampleTime, ...
       "inputType", inputType);

U = data.U;
Y = data.Y;

%% PLOT DATASET

figure

subplot(2,1,1)
plot(data.time,U)
ylabel("u(k)")
grid on

subplot(2,1,2)
plot(data.time,Y)
xlabel("Time")
ylabel("y(k)")
grid on

%% DATA PRE-PROCESSING

% Split train x test
[utr,uts,ytr,yts] = ...
    mltoolbox.preprocessing.train_test_split.split(U,Y,...
                                                   'train_ratio',train_ratio,...
                                                   'shuffle',shuffle);
% Time vetor - train and test
Ttr = data.time(1:size(ytr,1),:);
Tts = data.time(size(ytr,1)+1:end,:);

% Normalization
if normalize_input
    input_scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
    utr_norm = input_scaler.fit_transform(utr);
    uts_norm = input_scaler.transform(uts);
else
    utr_norm = utr;
    uts_norm = uts;
end

if normalize_output
    output_scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
    ytr_norm = output_scaler.fit_transform(ytr);
    yts_norm = output_scaler.transform(yts);
else
    ytr_norm = ytr;
    yts_norm = yts;
end

if normalize_input || normalize_output

    figure

    subplot(2,1,1)
    plot(Ttr,utr_norm)
    ylabel("u(k)")
    grid on

    subplot(2,1,2)
    plot(Ttr,ytr_norm)
    xlabel("Time")
    ylabel("y(k)")
    grid on

    figure

    subplot(2,1,1)
    plot(Tts,uts_norm)
    ylabel("u(k)")
    grid on

    subplot(2,1,2)
    plot(Tts,yts_norm)
    xlabel("Time")
    ylabel("y(k)")
    grid on

end

%% SYSTEM IDENTIFICATION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.systemId.OLSIdentifier('outputLag', 2, ...
                                         'inputLag', 2, ...
                                         'errorLag', 0, ...
                                         'includeCurrentInput', false, ...
                                         'approximation', approximation, ...
                                         'regularization', regularization);

model.fit(utr_norm,ytr_norm);
% 
% yhat_tr = model.predict(utr_norm,ytr_norm);
% yhat_ts = model.predict(uts_norm,yts_norm);

%% METRICS



%% END

















