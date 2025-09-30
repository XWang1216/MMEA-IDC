function [Populations, Re1, Re0, t,t1,NextStep, Non0_index, K] =  UpdateBackup(Populations,Re1, Re0, t,t1,NextStep, Non0_index, K,pop_size_K, Problem)
% Alg. 4
% calculate the current subpopulation size
pop_size = floor((Problem.N - pop_size_K)/K);
%% Update subpopulations (Intra-population Information Exchange)
for i = 1:K
    len = pop_size - size(Populations{i},2);
    % how to generate new solutions depending on the sparse distribution (Alg. 5)
    if len > 0 
        % using its own guiding vector
        P2   = sparse_gen(Re1{i}, max(1, len - floor(len/2)), Problem.D);
        % using all guiding vectors
        P1   = sparse_gen_D(i, Re1, max(1, floor(len/2)), Problem.D);
        % population reduction
        Populations{i} = [Populations{i},Problem.Evaluation(P1),Problem.Evaluation(P2)];
    end
    [Populations{i}, ~, ~] = EnvironmentalSelection_II([Populations{i}], pop_size);
end
%% Initializing a new complementary population (Inter-population Information Exchange)
for i = 1:K
    re1_mat(i,:) = Re1{i};
end
K = K + 1;
Re1{K} = sum(re1_mat,1) == 0;
Re0{K} = false(1, Problem.D);
t{K}   = 0;
t1{K}  = 1;
NextStep{K} = Problem.D;
Non0_index{K} = true(1, Problem.D);
re1 = cell(1,2);
re1{1} = Re1{K};
Populations{K} = Problem.Evaluation(sparse_gen_D(K-1, re1, pop_size_K, Problem.D));
end
