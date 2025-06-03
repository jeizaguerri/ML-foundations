function [distorsion] = distorsion(D, mu, c)
m = length(c);
i = (1:m);
muSel = mu(c, :);
dSel = D(i, :);
r =  dSel - muSel;
distancia = sum(r, 2);
distorsion = mean(distancia .^ 2);
%distorsion = sum(sqrt(sum((D - mu(c,:)).^2, 2)));

end

