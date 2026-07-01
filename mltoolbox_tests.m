%% TESTE DA BIBLIOTECA MLTOOLBOX

% Toolbox Initial Tests
% Author: David Nascimento Coelho
% Last mod: 2026/06/25

clear;
clc;

%% OPTIONS

which_dataset = 'loaded';
dims = [1 2 3];

%% GENERATE DATASET

rng(20)

N1 = 100;
N2 = 100;
N3 = 100;

X1 = randn(N1,2) + repmat([-2  0], N1, 1);
X2 = randn(N2,2) + repmat([ 2  0], N2, 1);
X3 = randn(N3,2) + repmat([ 0  2], N3, 1);

Xgen = [X1; X2; X3];
ygen = [ones(N1,1); 2*ones(N2,1); 3*ones(N3,1)];

%% LOAD DATASET

iris = mltoolbox.datasets.IrisDataset();
Xload = iris.X;
yload = iris.y;

%% CHOOSE / PLOT DATASET

switch which_dataset
    case "generated"
        X = Xgen;
        y = ygen;
        classNames = {};
    case "loaded"
        X = Xload;
        y = yload;
        classNames = iris.classes;
    otherwise
        X = Xgen;
        y = ygen;
end

mltoolbox.utils.plot_dataset(X,y,classNames,dims);

clear N1 N2 N3 X1 X2 X3 which_dataset iris Xgen ygen  Xload yload

%% DATA PRE-PROCESSING

% Embaralha
[X,y] = mltoolbox.preprocessing.shuffle_data(X,y);
mltoolbox.utils.plot_dataset(X,y,classNames,dims);

% Codifica Rótulos
%encoder = mltoolbox.preprocessing.LabelEncoder('mode','bipolar');
%y = encoder.fit_transform(y);

% Separa treino/teste
[Xtr,Xts,ytr,yts] = ...
    mltoolbox.preprocessing.train_test_split.split(X,y,'train_ratio',0.7);
mltoolbox.utils.plot_dataset([Xtr;Xts],[ytr;yts],classNames,dims);

% Normalização z-score
scaler = mltoolbox.preprocessing.DataScaler('mode','zscore');
Xtr_norm = scaler.fit_transform(Xtr);
Xts_norm = scaler.transform(Xts);
mltoolbox.utils.plot_dataset([Xtr_norm;Xts_norm],[ytr;yts],classNames,dims);

%% CLASSIFIER: LOAD / TRAIN / TEST

model = mltoolbox.classifiers.OLSClassifier();

model.fit(Xtr_norm,ytr);
yhat_tr = model.predict(Xtr_norm);
yhat_ts = model.predict(Xts_norm);

%% METRICS: ACC

acc_tr = mean(yhat_tr == ytr);
acc_ts = mean(yhat_ts == yts);

disp('Primeiras 10 previsões vs ground truth:')
disp([yhat_ts(1:10), yts(1:10)])

%% END


























