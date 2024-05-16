clear;clc;
load('train_sample.mat')
load('train_indx.mat')
seq= sample_20;
indexB = Bsample;
B = reshape(seq,[30600,2000]).';
b = B(:,1:end-200);
A_B = [b,indexB.'];
for i = 1:size(A_B, 2)
    C_B{i} = A_B(:, i);
end
data_B = table(C_B{:});
[idx_B,scores_f] = fscchi2(data_B,data_B(:,end));
idx_b = idx_B(1:4000);
idx_b(end+1)=30401;

pre_data_B = data_B(:,idx_b);
cv = cvpartition(size(pre_data_B,1),'HoldOut',0.25);
test = cv.test;
training_data_real = [real((pre_data_B{~test,1:end-1})), imag(pre_data_B{~test,1:end-1})];
testing_data_real = [real(pre_data_B{test,1:end-1}), imag(pre_data_B{test,1:end-1})];

B_train = table2array(pre_data_B(~test,end));
num_trees = 500;
model_B = TreeBagger(num_trees, training_data_real, B_train);

save("ML_model_B.mat","model_B")
save("features_selected_B.mat","idx_B")