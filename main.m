%% load utils and MNIST data
clc; clear;
disp('reading data');tic;
addpath('./utils');
[train, test] = DataPrep('./data');
toc;

disp('initializing data');tic;
D = 400; %% dimension of each multinomial distribution (same as the number of pixels in each image)
K = 40;  %% number of motinomial distributions
pi = ones (K, 1) / K;
mu = 0.25 + 0.5 * rand (D, K);
N = size (train.images, 2);
gamma = zeros (N, K);
count = 0;
pi_new = zeros (K, 1);
toc;

while (1)
    disp ('E step'); tic;
    for i = 1:N
        gamma (i, :) = multinomial (train.images (:, i), mu, pi);
        gamma (i, :) = gamma (i, :) ./ sum (gamma (i, :));
    end
    toc;
    disp ('M step'); tic;
    mu_new = zeros (D, K);
    pi_new = zeros (K, 1);
    for i = 1:K
        Nk = sum (gamma (:, i));
        for j = 1:N
            mu_new(:, i) = mu_new(:, i) + gamma (j, i) * train.images (:, j);
        end
        mu_new(:, i) = mu_new(:, i) / Nk;
        pi_new (i) = Nk / N;
    end
    count = count + 1;toc;
    sprintf( '|pi - pi_new| = %d, log likelihood: %d', norm (pi - pi_new), loglikelihood(train.images, mu_new, pi))
    if (norm (pi - pi_new) <= 0.01)
        break;
    end
    mu = mu_new;
    pi = pi_new;
end
outfile = strcat ('trainedPara', num2str(K), '.mat');
save (outfile, 'mu', 'pi');
ShowModel(mu, pi, 5, 8, 1:K);