%% UNIT TEST

clear
clc

%% DATASET

rng(1)

N1 = 100;
N2 = 100;
N3 = 100;

X1 = randn(N1,2) + repmat([-2  0], N1, 1);
X2 = randn(N2,2) + repmat([ 2  0], N2, 1);
X3 = randn(N3,2) + repmat([ 0  2], N3, 1);

X = [X1; X2; X3];

Y = [ones(N1,1); 2*ones(N2,1); 3*ones(N3,1)];

%% INIT MODEL

mdl_1 = lmsClassifier2();
mdl_1.learningRate = 0.05;
mdl_1.numEpochs = 200;
mdl_1.addBias = true;
mdl_1.shuffleEachEpoch = true;

mdl_1 = mdl_1.fit(X, Y);

mdl2 = lmsClassifier2( ...
       'learningRate',0.05,...
       'numEpochs',200,...
       'addBias', true, ...
       'shuffleEachEpoch', true);

mdl2 = mdl2.fit(X, Y);

yhat = mdl_1.predict(X);

acc = mdl_1.score(X, Y);

disp("accuracy: ")
disp(acc)

%% END