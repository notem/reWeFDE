function MI_est = MutualInfo(data)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FeatureSample1 = data(1,:);
bandwidth1 = 0.9 * min( std(FeatureSample1), iqr(FeatureSample1)/1.34 ) * length(FeatureSample1)^(-0.2);
if bandwidth1 == 0
    bandwidth1 = 0.1;
end

FeatureSample2 = data(2,:);
bandwidth2 = 0.9 * min( std(FeatureSample2), iqr(FeatureSample2)/1.34 ) * length(FeatureSample2)^(-0.2);
if bandwidth2 == 0
    bandwidth2 = 0.1;
end

p = kde(data, [bandwidth1; bandwidth2]);
p1 = marginal(p,[1]);
p2 = marginal(p,[2]);
% MI(a,b) = H(a) + H(b) - H(a,b)

MI_est = entropy(p1)+entropy(p2)-entropy(p);

