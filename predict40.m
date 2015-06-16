disp('reading data');
addpath('./utils');
[train, test] = DataPrep('./data');

disp('initializing data');
load('trainedPara.mat');
D = 400; %% dimension of each multinomial distribution (same as the number of pixels in each image)
K = size (mu, 2);  %% number of motinomial distributions

classCount = zeros (10, 1);
N = 50000;
if 0
m = zeros (K, 10);
disp('training parameters');
for i=1:D
    for j = 1:K
        if mu(i,j) == 0
            mu(i,j) = 10^-6;
        end
    end
end
for i=1:N
    label = train.labels (i) + 1;
    classCount(label) = classCount(label) + 1;
    m(:, label) = m(:, label) + pi(label) * multinomial( train.images(:, i), mu, pi);
end
for i=1:10
    m (:, i) = m (:, i) / classCount(i);
end
end

testCount = length (test.labels);
correctCount = 0.0;
p = zeros (K, 1);
l = zeros (10,1);
for i = 1:testCount
    p = multinomial (test.images(:, i), mu, pi);
    for j = 1:10
        l(j) = norm (p - m(:,j));
    end
    [label, label] = min (l);
    if label == test.labels (i) + 1
        correctCount = correctCount + 1;
    else
        sprintf ('wrong %d -> %d', test.labels (i), label-1)
    end
end
100*correctCount/testCount

