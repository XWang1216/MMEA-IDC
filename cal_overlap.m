function [merge_inx] = cal_overlap(Re1, K, stand_r)
re1_mat = [];
for i = 1:K
    re1_mat(i,:) = Re1{i};
end
merge_inx = [];
num_ind = 1;
for i =1:K
    for j = i+1:K
        overlap_r(i,j) = sum(re1_mat(i,:) .* re1_mat(j,:))/min(sum(re1_mat(i,:)), sum(re1_mat(j,:)));
        if overlap_r(i,j) > stand_r
            merge_inx(num_ind,:) = [i,j];
            num_ind = num_ind + 1;
        end
    end
end
end