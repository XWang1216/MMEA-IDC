function [population]  = sparse_gen(Re1, N, D)
population = zeros(N,D);
population_temp = UniformPoint(N,D,'Latin');
population(:,Re1) = population_temp(:,Re1);
end