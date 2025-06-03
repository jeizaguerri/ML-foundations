function Z = updateClusters(D,mu)
% D(m,n), m datapoints, n dimensions
% mu(K,n) final centroids
%
% c(m) assignment of each datapoint to a class

% Calcular distancias a cada centroide para cada dato
k = size(mu,1);
distancias = zeros(size(D,1), k);
for i = (1:k)
    r = D - mu(i, :);
    distancias(:,i) = sum(r.^2, 2);
end
[~, Z] = min(distancias, [], 2);