%% EXAMPLE OF A SYSTEM IDENTIFICATION EXPERIMENT

% Machine Learnning Toolbox
% Last mod: 2026/07/14

clear;
clc;

%% OPTIONS

% Data Options
dataName = "arxmimo"; % firstordersystem secondordersystem firsystem arx arxmimo statespacesystem nonlinearnarx hammersteinsystem wienersystem
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
outputLags = [2 2];
inputLags = [2 2];
errorLags = [];
includeCurrentInput = true;
stepsAhead = 1;
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

% figure
% 
% subplot(2,1,1)
% plot(data.time,U)
% ylabel("u(k)")
% grid on
% 
% subplot(2,1,2)
% plot(data.time,Y)
% xlabel("Time")
% ylabel("y(k)")
% grid on

%% DATA PRE-PROCESSING

% Split train x test
[Utr,Uts,Ytr,Yts] = ...
    mltoolbox.preprocessing.train_test_split(U,Y,...
                                            'train_ratio',train_ratio,...
                                            'shuffle',shuffle);
% Time vetor - train and test
Ttr = data.time(1:size(Ytr,1),:);
Tts = data.time(size(Ytr,1)+1:end,:);

% Normalization
if normalize_input
    input_scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
    Utr_norm = input_scaler.fit_transform(Utr);
    Uts_norm = input_scaler.transform(Uts);
else
    Utr_norm = Utr;
    Uts_norm = Uts;
end

if normalize_output
    output_scaler = mltoolbox.preprocessing.DataScaler('mode',normalization);
    Ytr_norm = output_scaler.fit_transform(Ytr);
    Yts_norm = output_scaler.transform(Yts);
else
    Ytr_norm = Ytr;
    Yts_norm = Yts;
end

if normalize_input || normalize_output

    figure

    subplot(2,1,1)
    plot(Ttr,Utr_norm)
    ylabel("u(k)")
    grid on

    subplot(2,1,2)
    plot(Ttr,Ytr_norm)
    xlabel("Time")
    ylabel("y(k)")
    grid on

    figure

    subplot(2,1,1)
    plot(Tts,Uts_norm)
    ylabel("u(k)")
    grid on

    subplot(2,1,2)
    plot(Tts,Yts_norm)
    xlabel("Time")
    ylabel("y(k)")
    grid on

end

%% SYSTEM IDENTIFICATION MODEL: LOAD / TRAIN / TEST

model = mltoolbox.systemId.OLSIdentifier('outputLags', outputLags, ...
                                         'inputLags', inputLags, ...
                                         'errorLags', errorLags, ...
                                         'includeCurrentInput', includeCurrentInput, ...
                                         'stepsAhead',stepsAhead, ...
                                         'approximation', approximation, ...
                                         'regularization', regularization);

model.fit(Utr_norm,Ytr_norm);

yhat_ts = model.predict(Uts_norm,Yts_norm);

figure;
plot(Yts_norm(:,1),yhat_ts(:,1),'b.');
title('Yts x Yts-hat - Must be a line')
xlabel('Yts')
ylabel('Yts-hat')

% ToDo - Clear Memory
% yhat_tr = model.predict(utr_norm,ytr_norm);

%% METRICS



%% END

















