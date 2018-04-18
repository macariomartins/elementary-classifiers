%
% IRIS LOG CLASSIFICATION
%

%% Cleaning any mess
% Commands to clean the Matlab workspace
close all;
clear;
clc;

%% Loading the Database
% Loads the iris_log.dat file and prepare the Base and Classes informations
load('database\iris_log.dat');
Base = iris_log(:, 1:4);
Classes = iris_log(:, 5:7);
clear iris_log;

%% Settings for K-NN (K-Nearest Neighbour) Classifier
% First argument is K value, the second is the distance method that will be
% used to calculate distances; and the last argument is a boolean value in-
% dicating whether z-score must be used (true) or not (false).
KNN_1 = classifiers.KNN_Classifier(1, "euclidean", false);
KNN_2 = classifiers.KNN_Classifier(1, "euclidean", true);

%% Settings for NCC (Nearest Centroid Classifier) Classifier
% First argument is the distance method that will be used to calculate dis-
% tances; and the last argument is a boolean value indicating whether
% z-score must be used (true) or not (false).
NCC_1 = classifiers.NCC_Classifier("euclidean", false);
NCC_2 = classifiers.NCC_Classifier("euclidean", true);

%% Settings for Hold-Out validation
% First argument is the number of trials we want to run with HoldOut vali-
% dation method, the second is the database percentage that must be used to
% test de classifications.
HoldOut = validations.HoldOut_Validation(20, 0.3);

%% Settings for Leave-One-Out validation
% You can set a number of trials by inserting an integer argument to the
% constructor bellow
LOO = validations.LeaveOneOut_Validation(1);

%% Running...
% The lines bellow show the mean accuracy of each Classifier used with a
% corresponding validation method. Results are sent to command window.
fprintf("Acurácias para o classificador 1-NN (dist. Euclidiana)\n");
fprintf("------------------------------------------------------\n");
fprintf("Hold Out (20 rodadas) sem Z-Score: %f\n",    HoldOut.calculateAccuracy(KNN_1, Base, Classes));
fprintf("Hold Out (20 rodadas) com Z-Score: %f\n",    HoldOut.calculateAccuracy(KNN_2, Base, Classes));
fprintf("Leave-One-Out (1 rodada) sem Z-Score: %f\n", LOO.calculateAccuracy(KNN_1, Base, Classes));
fprintf("Leave-One-Out (1 rodada) com Z-Score: %f\n", LOO.calculateAccuracy(KNN_2, Base, Classes));
fprintf("\n");
fprintf("Acurácias para o classificador NCC (dist. Euclidiana)\n");
fprintf("-----------------------------------------------------\n");
fprintf("Hold Out (20 rodadas) sem Z-Score: %f\n",    HoldOut.calculateAccuracy(KNN_1, Base, Classes));
fprintf("Hold Out (20 rodadas) com Z-Score: %f\n",    HoldOut.calculateAccuracy(KNN_2, Base, Classes));
fprintf("Leave-One-Out (1 rodada) sem Z-Score: %f\n", LOO.calculateAccuracy(KNN_1, Base, Classes));
fprintf("Leave-One-Out (1 rodada) com Z-Score: %f\n", LOO.calculateAccuracy(KNN_2, Base, Classes));
