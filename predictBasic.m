%% load utils and MNIST data
clc; clear;
disp('reading data');
addpath('./utils');
[train, test] = DataPrep('./data');

disp('initializing data');
D = 400; %% dimension of each multinomial distribution (same as the number of pixels in each image)
K = 10;  %% number of motinomial distributions
pi = zeros (K, 1);
mu = zeros (D, K);
N = 50000;

disp('training parameters');
for i=1:N
    label = train.labels (i) + 1;
    pi (label) = pi (label) + 1;
    mu (:, label) = mu (:, label) + train.images (:, i);
end
for i=1:K
    mu (:, i) = mu (:, i) / pi(i) + 0.000001;
end
pi = pi / N;

mu_plot = reshape(mu, 20, 20, length(pi));
ShowModel (mu_plot, pi, 2, 5, 1:K);

testCount = length (test.labels);
correctCount = 0.0;
p = zeros (K, 1);
for i = 1:testCount
    x = test.images(:, i);
    for j = 1:K
        
        p(j) = sum( (x .* log (mu (:, j))) + ((1-x) .* log(1-mu(:, j))));
    end
    [label, label] = max (p);
    if label == test.labels (i) + 1
        correctCount = correctCount + 1;
    end
end
100*correctCount/testCount
