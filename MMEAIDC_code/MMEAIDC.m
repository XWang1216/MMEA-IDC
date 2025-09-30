classdef MMEAIDC < ALGORITHM
    % <multi> <real> <large/none> <multimodal> <sparse/none>
    % size-imbalanced dual population framework with complementary search
    %------------------------------- Reference --------------------------------
    % X. Wang and Y. Jin, A Size-Imbalanced Dual Population with Complementary 
    % Search for Sparse Large-Scale Multi-Modal Multi-Objective
    % Optimization, IEEE Transactions on Evolutionary Computation, 2025
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate two populations and their non-zero mask vectors
            % pre-defined subpopulation and complementary population size
            pop_size_K = 20; 
            Mask = zeros(Problem.N - pop_size_K, Problem.D);  
            Population    = Problem.Evaluation(Mask);
            % subpopulations in the main population
            K = ceil((Problem.N - pop_size_K)/20); 
            Populations = cell(1,K);
            Re0      = cell(1,K);
            Re1      = cell(1,K);
            t        = cell(1,K);
            t1       = cell(1,K);
            NextStep = cell(1,K);
            Non0_index  = cell(1,K);
            index       = randperm(floor((Problem.N - pop_size_K)/K)*K);
            temp        = reshape(index,K,floor((Problem.N - pop_size_K)/K));
            for i = 1 : K
                Populations{i} = Population(temp(i,:));
                [Populations{i},  ~, ~] = EnvironmentalSelection_II(Populations{i},floor((Problem.N - pop_size_K)/K));
                Re0{i} = false(1, Problem.D);
                Re1{i} = true(1, Problem.D);
                t{i}   = 0;
                t1{i}  = 1;
                NextStep{i} = Problem.D;
                Non0_index{i} = true(1, Problem.D);
            end
            % complementary population
            K = K + 1;
            Re0{K} = false(1, Problem.D);
            Re1{K} = true(1, Problem.D);
            t{K}   = 0;
            t1{K}  = 1;
            NextStep{K} = Problem.D;
            Non0_index{K} = true(1, Problem.D);
            Populations{K} = Problem.Evaluation(sparse_gen_D(1, Re1, pop_size_K, Problem.D));
            [Populations{K},  ~, ~] = EnvironmentalSelection_II(Populations{K}, pop_size_K);
            K_All = zeros(1, floor(Problem.maxFE/Problem.N));
            iter = 1;
            Merge_iter = [1];

            %% Optimization
            evaluation = 0;
            intervel = floor(Problem.maxFE / Problem.N / 5);
            while Algorithm.NotTerminated(Population)
                K_All(iter) = K;
                iter = iter + 1;
                evaluation = evaluation + 1;
                %% Adaptive Clustering-Enhanced GA (independent update each subpopulation and complementary population, Alg. 2)
                for i = 1:K
                    NextStep{i} = mean(abs(Re1{i}-Re0{i})) >= (1/Problem.D);
                    % exploration
                    if NextStep{i}
                        MatingPool = TournamentSelection(2,size(Populations{i},2),sum(max(0,Populations{i}.cons),2));
                        % FE is a multiple of T?
                        if mod(evaluation,intervel) == 0 
                            t{i} = t{i} + 1;
                            if t{i} > 1
                                Re0{i} = Re1{i};
                            end
                            [Re1{i}, PopulationM] = DimJud(Populations{i}(MatingPool).decs, Problem.upper, Problem.lower, Re0{i}); % ECS conducted
                        else
                            PopulationM = Populations{i}(MatingPool).decs;
                        end
                        Offspring = OperatorGA(Problem, PopulationM, {1,20,1,1});
                        Offspring = Problem.Evaluation(Offspring);
                        [Populations{i},  ~, ~]  = EnvironmentalSelection_II([Populations{i},Offspring],size(Populations{i},2));
                    % Exploitation
                    else
                        MatingPool = TournamentSelection(2,size(Populations{i},2),sum(max(0,Populations{i}.cons),2));
                        Offspring0 = Populations{i}(MatingPool).decs;
                        % FE is a multiple of T?
                        if t1{i} == 1 || mod(evaluation,intervel) == 0
                            [Offspring,Non0_index{i}] = DimJud0(Offspring0, Problem); % ECS conducted
                            t1{i} = t1{i} + 1;
                        else
                            flag = true;
                            Offspring = GA0(Offspring0, Non0_index{i}, Problem, flag); % GA with high mutation
                        end
                        Offspring = Problem.Evaluation(Offspring);
                        [Populations{i},  ~, ~]  = EnvironmentalSelection_II([Populations{i},Offspring],size(Populations{i},2));
                    end
                end

                %%  Mutual Population Update (co-evolution, Alg. 4) 
                % Only if FE is a multiple of T
                if mod(evaluation,intervel) == 0 && K > 2  &&  evaluation < floor(Problem.maxFE / Problem.N * 0.95)
                    % merging-only
                    stand_r = 0.5;
                    merge_inx = cal_overlap(Re1, K-1, stand_r);
                    if size(merge_inx,1)>0
                        for l = size(merge_inx,1): 1
                            i = merge_inx(l,1);
                            j = merge_inx(l,2);
                            [Populations{i}] = [Populations{i},Populations{j}];
                            if sum(Re1{i}) > sum(Re1{j}) 
                                Re1{i} = Re1{j};
                            end
                        end
                        Populations(merge_inx(:,2)) = [];
                        Re0(merge_inx(:,2)) = [];
                        Re1(merge_inx(:,2)) = [];
                        t(merge_inx(:,2)) = [];
                        t1(merge_inx(:,2)) = [];
                        NextStep(merge_inx(:,2)) = [];
                        Non0_index(merge_inx(:,2)) = [];
                    end
                    % Information exchange
                    [~,best]     = SubPopRank(Populations);
                    [~, idx]     = max(best);
                    Flag = idx < size(best,2) || best(end)<=3; 
                    if Flag % the CP should be transferred 
                        K = size(t,2); 
                        [Populations, Re1, Re0, t,t1,NextStep, Non0_index, K] = UpdateBackup(Populations, Re1, Re0, t,t1,NextStep, Non0_index, K, pop_size_K, Problem);
                    else % the CP should not be transferred
                         K = size(t,2) - 1;
                        [Populations, Re1, Re0,t,t1,NextStep, Non0_index, K] = UpdateBackup(Populations,Re1, Re0, t,t1,NextStep, Non0_index, K, pop_size_K, Problem);
                    end
                end
                %% Save the number of subpopulations trend over evolutionary process
                Population = [Populations{:}];
                if Problem.FE >= Problem.maxFE
                    folder = fullfile('Data',class(Algorithm));
                    [~,~]  = mkdir(folder);
                    file   = fullfile(folder,sprintf('%s_%s_M%d_D%d_K_',class(Algorithm),class(Problem),Problem.M,Problem.D));
                    runNo  = 1;
                    while exist([file,num2str(runNo),'.mat'],'file') == 2
                        runNo = runNo + 1;
                    end
                    save([file,num2str(runNo),'.mat'],'K_All','Merge_iter');
                end
            end
        end
    end
end