clear;clc;
load('train_sample.mat')
load('train_indx.mat')
seq= sample_20;
indexf = fsample;
indexB = Bsample;
B = reshape(seq,[30600,2000]).';
b = B(:,1:end-200);
A_f = [b,indexf.'];
for i = 1:size(A_f, 2)
    C_f{i} = A_f(:, i);
end
data_f = table(C_f{:});
[idx_f,scores_f] = fscchi2(data_f,data_f(:,end));
idx = idx_f(1:5000);
idx(end+1)=30401;
pre_data_f = data_f(:,idx);
cv = cvpartition(size(pre_data_f,1),'HoldOut',0.25);
test = cv.test;
training_data_real = [real((pre_data_f{~test,1:end-1})), imag(pre_data_f{~test,1:end-1})];
testing_data_real = [real(pre_data_f{test,1:end-1}), imag(pre_data_f{test,1:end-1})];

f_train = table2array(pre_data_f(~test,end));
num_trees = 500;
model = TreeBagger(num_trees, training_data_real, f_train);

predicted_f =(predict(model, testing_data_real));
f_20_t = table2cell(pre_data_f(test,end));

save("ML_model_f.mat","model")

save("features_selected_f.mat",'idx_f')