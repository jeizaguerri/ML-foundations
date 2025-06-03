function [mu, c] = kmeans(D,mu0)

% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

lastC = zeros(size(D,1), 1);
c  = ones(size(D,1), 1);
mu = mu0;
while ~isequal(c, lastC)
    lastC = c;
    c  = updateClusters(D, mu);
    mu = updateCentroids(D, c);
end

end