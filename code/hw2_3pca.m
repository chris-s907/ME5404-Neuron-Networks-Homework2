clc
clear
set(0,'defaultfigurecolor','w')
%%
train_samples_filepath = 'F:\matlab\bin\neuron networks\group_1\train\';
test_samples_filepath = 'F:\matlab\bin\neuron networks\group_1\test\';
train_img_files = dir([train_samples_filepath '*.jpg']);
test_img_files = dir([test_samples_filepath '*.jpg']);
train_num = length(train_img_files);
test_num = length(test_img_files);

train_label = zeros(1,train_num);
test_label = zeros(1,test_num);
train_img = zeros(256*256,train_num);
test_img = zeros(256*256,test_num);

for i = 1:train_num
    filename = [train_samples_filepath train_img_files(i).name];
    label = str2double(filename(50));
    train_label(1,i) = label;
    I = imread(filename);
    train_img(:,i) = I(:); %convert the matrix into a column vector
end


for i = 1:test_num
    filename = [test_samples_filepath test_img_files(i).name];
    label = str2double(filename(49));
    test_label(1,i) = label;
    I = imread(filename);
    test_img(:,i) = I(:); %convert the matrix into a column vector
end

all_img = [train_img,test_img];
all_label = [train_label, test_label];
[coeff,score,latent,tsquared] = pca(all_img');
acu = cumsum(latent)./sum(latent);
acu_index = size(acu,1);
for i = 1:size(acu,1)
    if acu(i) > 0.99
        acu_index = i;
        break
    end
end
all_img_pca = score(:,1:acu_index)';
train_img = all_img_pca(:,1:size(train_img,2));
test_img = all_img_pca(:,size(train_img,2)+1:size(all_img,2));

%% Rosenblatt's perceptron
%learning rate
eta1 = 0.1;
episode = 1000;
%initial value
X_train = [ones(1,503);
           train_img];
X_test = [ones(1,167);
           test_img];
d_train = train_label;
d_test = test_label;
w = randi([-5 5], 1, size(train_img,1)+1);
e = zeros(1,503);
train_acc = zeros(1,episode);
test_acc = zeros(1,episode);

% learning algorithm
for n = 1: episode
    y_train = w * X_train;
    for i = 1:503
        if y_train(1,i) > 0
           y_train(1,i) = 1;
        else if y_train(1,i) < 0
                y_train(1,i) = 0;
            end
        end   
    end
    e = d_train - y_train;
    w = w + eta1 * e * X_train';
    num_same_train = sum(y_train == d_train);
    train_acc(n) = num_same_train/503;
    
    y_test = w * X_test;
    for i = 1:167
        if y_test(1,i) > 0
           y_test(1,i) = 1;
        else if y_test(1,i) < 0
                y_test(1,i) = 0;
            end
        end   
    end
    num_same_test =  sum(y_test == d_test);
    test_acc(n) = num_same_test/167;
end

plot(train_acc);
hold on
plot(test_acc);
legend('train accuracy','test accuracy');
title('pca');