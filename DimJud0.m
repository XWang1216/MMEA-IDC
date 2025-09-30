function [Parent, Non0_index]= DimJud0(Parent,Problem)
%% ECS
[~,N2] =  size(Parent);
density = mean(Parent~=0,1);
for i = 1: N2
    dim_Q75(i) = prctile(Parent(Parent(:,i)~=0,i),75);
    dim_Q50(i) = prctile(Parent(Parent(:,i)~=0,i),50);
    dim_Q25(i) = prctile(Parent(Parent(:,i)~=0,i),25);
end
dim_Q75(isnan(dim_Q75)) = 0;
dim_Q50(isnan(dim_Q50)) = 0;
dim_Q25(isnan(dim_Q25)) = 0;
ParentN = zeros(1, N2);
upper_index = dim_Q50>0;
ParentN(upper_index) = dim_Q75(upper_index)./Problem.upper(upper_index);
lower_index = dim_Q50<0;
ParentN(lower_index) = dim_Q25(lower_index)./Problem.lower(lower_index);
input_feature = ParentN .* density;
[C_indx,C_point] = ExactKmeans(input_feature' , 2);
[~,a] = sort(C_point(:,1));
Is0_index1 = C_indx == a(1);
Is0_index = Is0_index1  == 1;
Non0_index = 1-Is0_index;
%% setting 0
Parent(:, Is0_index) = 0;
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