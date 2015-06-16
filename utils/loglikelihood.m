function [ p ] = loglikelihood( x, mu, pi )
N = size (x, 2);
p = 0;
for i=1:N
    p = p + log (sum (multinomial (x(:,i), mu, pi)));
end
end

