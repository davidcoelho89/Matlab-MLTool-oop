%% EXAMPLE OF A REGRESSION EXPERIMENT

% Machine Learnning Toolbox
% Author: David Nascimento Coelho
% Last mod: 2026/07/02

clear;
clc;

%% OPTIONS

dataName = "LinearRegression"; 	% SincRegression ; SinRegression ; FrankeFunction
                                % LinearRegression ; MultipleLinearRegression ; 
                                % PolynomialRegression ; 
number_of_samples = 200;
noise_std = 0.10;
randomState = 10;

%% LOAD DATASET

data = mltoolbox.datasets.ArtificialRegressionDataset(dataName, ...
                                                      "nSamples", number_of_samples, ...
                                                      "noiseStd", noise_std, ...
                                                      "randomState", randomState);
X = data.X;
Y = data.Y;

%% PLOT DATASET

figure;
plot(X,Y,'.');

%% END















