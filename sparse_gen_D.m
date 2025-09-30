function [population]  = sparse_gen_D(ind, Re1, N, D) % re1=1 indicates the dimension that is already activated 
for i = 1:size(Re1,2)-1
    re1_mat(i,:) = Re1{i};
end
re1_sum = sum(re1_mat,1) ~= 0; % indicted that all the dimensions that is already activated among all subpopulations
re1_sum_rp = repmat(re1_sum, N, 1);
mu = rand(N,D);
temp = logical(1 - (re1_sum_rp & mu<=0.5));
population = UniformPoint(N,D,'Latin');
population(temp) = 0;
end