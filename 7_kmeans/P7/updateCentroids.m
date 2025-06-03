function munew = updateCentroids(D,c)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

K = max(c);
for i = (1:K)
    munew(i,:) = mean(D(c==i, :));
end
