function [p] = multinomial (x, u, pi)
    assert (length(x)  == size(u, 1), 'size mismatch');
    assert (length(pi) == size(u, 2), 'size mismatch');
    K = size  (u, 2);
    p = zeros (K, 1);
    for i = 1:K
        p(i) = pi(i) * prod( (u(:, i).^x) .* ((1-u(:, i)).^(1-x)) );
    end
end

