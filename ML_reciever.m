clear;clc;
load ML_model_f.mat
load samples.mat
load indices.mat
load features_selected_f.mat
load ML_model_B.mat
load features_selected_B.mat
%%
tic;
data_20 = data(sample_20);
data_10 = data(sample_10);
data_0 = data(sample_0);
init_f_10 = dif(data_10,model,idx_f);
init_f_20 = dif(data_20,model,idx_f);
init_f_0 = dif(data_0,model,idx_f);
bw_20 = bw(data_20,model_B,idx_B);
bw_10 = bw(data_10,model_B,idx_B);
bw_0 = bw(data_0,model_B,idx_B);
toc;
save("test.mat","init_f_10","init_f_20","init_f_0","bw_20","bw_10","bw_0")
%%
%load indices.mat
%load test.mat
%% cell to double
clc;
bw_20 = cellfun(@str2double,bw_20);
bw_10 = cellfun(@str2double,bw_10);
bw_0 = cellfun(@str2double,bw_0);
init_f_20 = cellfun(@str2double,init_f_20);
init_f_10 = cellfun(@str2double,init_f_10);
init_f_0 = cellfun(@str2double,init_f_0);
%%
Cb20 = confusionmat(Bsample,bw_20);
acc_b20 = sum(diag(Cb20))/sum(Cb20(:));
Cb10 = confusionmat(Bsample,bw_10);
acc_b10 = sum(diag(Cb10))/sum(Cb10(:));
Cb0 = confusionmat(Bsample,bw_0);
acc_b0 = sum(diag(Cb0))/sum(Cb0(:));
Cf20 = confusionmat(fsample,init_f_20);
acc_f20 = sum(diag(Cf20))/sum(Cf20(:));
Cf10 = confusionmat(fsample,init_f_10);
acc_f10 = sum(diag(Cf10))/sum(Cf10(:));
Cf0 = confusionmat(fsample,init_f_0);
acc_f0 = sum(diag(Cf0))/sum(Cf0(:));
%%
t = [ acc_f0, acc_f10, acc_f20; acc_b0, acc_b10, acc_b20];
row_names = {'precision of initial frequency', 'precision of bandwidth'};
col_names = {'0 dB', '10 dB', '20 dB'};
T = array2table(t, 'RowNames', row_names, 'VariableNames', col_names);
save("presicion table.mat","T")

function detect_B = bw(data,model,idx_B)
    idx = idx_B(1:4000);
    pre_data_B = data(:,idx);
    data_real = [real((pre_data_B{:,:})), imag(pre_data_B{:,:})];

    detect_B = predict(model,data_real);
end


function detect_init_f = dif(data,model,idx_f)
    idx = idx_f(1:5000);
    pre_data_f = data(:,idx);
    data_real = [real((pre_data_f{:,:})), imag(pre_data_f{:,:})];

    detect_init_f = predict(model,data_real);
end
function data = data(sequence)
    w = size(sequence,2);
    B = reshape(sequence,[30600,w/30600]).';
    b = B(:,1:end-200);
    for i = 1:size(b, 2)
       C{i} = b(:, i);
    end
    data = table(C{:});
end