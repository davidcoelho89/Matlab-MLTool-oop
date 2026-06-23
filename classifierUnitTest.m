%% UNIT TEST

clear
clc

%% GENERATE DATASET

rng(1)

N1 = 100;
N2 = 100;
N3 = 100;

X1 = randn(N1,2) + repmat([-2  0], N1, 1);
X2 = randn(N2,2) + repmat([ 2  0], N2, 1);
X3 = randn(N3,2) + repmat([ 0  2], N3, 1);

X = [X1; X2; X3];

Y = [ones(N1,1); 2*ones(N2,1); 3*ones(N3,1)];

%% PLOT DATASET

figure; hold on; grid on;

classes = unique(Y);
colors = lines(length(classes));

for i = 1:length(classes)
    idx = (Y == classes(i));

    plot(X(idx,1), X(idx,2), 'o', ...
        'Color', colors(i,:), ...
        'MarkerFaceColor', colors(i,:), ...
        'LineStyle', 'none', ...
        'MarkerSize', 3);
end

xlabel('X_1');
ylabel('X_2');
title('Dataset 2D por classe (plot)');
legend("Classe " + string(classes));

%% EXPERIMENT WITH 1 REALIZATION  - SETTING PARAMETERS AFTER CONSTRUCTOR

mdl_1 = lmsClassifier2();
mdl_1.learningRate = 0.05;
mdl_1.numEpochs = 200;
mdl_1.addBias = true;
mdl_1.shuffleEachEpoch = true;

mdl_1 = mdl_1.fit(X, Y);
yhat = mdl_1.predict(X);
acc = mdl_1.score(X, Y);

disp("accuracy 1:")
disp(acc)

%% EXPERIMENT WITH 1 REALIZATION  - SETTING PARAMETERS WITH CONSTRUCTOR

mdl_2 = lmsClassifier2( ...
       'learningRate',0.05,...
       'numEpochs',200,...
       'addBias', true, ...
       'shuffleEachEpoch', true);

mdl_2 = mdl_2.fit(X, Y);
yhat = mdl_2.predict(X);
acc = mdl_2.score(X, Y);

disp("accuracy 2: ")
disp(acc)

%% END