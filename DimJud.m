function [Re, Population] = DimJud(Population, upper, lower,ReAlready)
%% ECS
[N1,N2] =  size(Population);
density = mean(Population~=0,1);
for i = 1: N2
    dim_Q75(i) = prctile(Population(Population(:,i)~=0,i),75);
    dim_Q50(i) = prctile(Population(Population(:,i)~=0,i),50);
    dim_Q25(i) = prctile(Population(Population(:,i)~=0,i),25);
end
dim_Q75(isnan(dim_Q75)) = 0;
dim_Q50(isnan(dim_Q50)) = 0;
dim_Q25(isnan(dim_Q25)) = 0;
km_input = zeros(1,N2);
upper_index = dim_Q50>0;
km_input(upper_index) = dim_Q75(upper_index)./upper(upper_index);
lower_index = dim_Q50<0;
km_input(lower_index) = dim_Q25(lower_index)./lower(lower_index);
input_feature = km_input .* density;
[C_indx,C_point] = ExactKmeans(input_feature', 2);
[~,a] = max(C_point(:,1));
activating_index = C_indx == a;

%% Reinitializing Non-Zero Prone Variables
% Case 1: dim_mean > 0
ReNow = false(1,size(ReAlready,2));
upper_index1 = upper_index + activating_index;
ReNow(upper_index1==2) = true;
ReNow(ReAlready == true) = false;
range=[];
range =  upper(ReNow) - dim_Q75(ReNow);
if ~isempty(range)
    Population(:,ReNow) = rand(N1,size(range,2)).*range + repmat(dim_Q75(ReNow),N1,1);
end
ReAlready(ReNow ==true) = true;

% Case 2: dim_mean < 0
upper_index1 =  lower_index + activating_index;
ReNow(upper_index1==2) = true;
ReNow(ReAlready ==true) = false;
range=[];
range =  dim_Q25(ReNow) - lower(ReNow);
if ~isempty(range)
    Population(:,ReNow) = -1.*rand(N1,size(range,2)).*range + repmat(dim_Q25(ReNow),N1,1);
end
ReAlready(ReNow ==true) = true;
Re = ReAlready;
end

function [c_indx,c_point] = ExactKmeans(feature, K)
[N1, N2] = size(feature);
% Finding the max and min value
[f_max] = max(feature);
[f_min] = min(feature);

A1 = f_max;
B1 = f_min;
A = A1;
B = B1;
for i = 1:100
    dis(:,1) = sqrt((feature(:,1)-A).^2);
    dis(:,2) = sqrt((feature(:,1)-B).^2);
    for j = 1:N1
        [~,indx(j)] = min(dis(j,:));
    end
    A = mean(feature(indx==1,:),1);
    B = mean(feature(indx==2,:),1);
    A(isnan(A)) = A1(isnan(A));
    B(isnan(B)) = B1(isnan(B));
   
end
c_indx = indx;
c_point = [A;B];
end