clc
clear
set(0,'defaultfigurecolor','w')
%% image processing
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

%% Rosenblatt's perceptron
% %learning rate
% eta1 = 0.1;
% episode = 800;
% %initial value
% X_train = [ones(1,503);
%            train_img];
% X_test = [ones(1,167);
%            test_img];
% d_train = train_label;
% d_test = test_label;
% w = randi([-5 5], 1, 65537);
% e = zeros(1,503);
% train_acc = zeros(1,episode);
% test_acc = zeros(1,episode);
% 
% % learning algorithm
% for n = 1: episode
%     y_train = w * X_train;
%     for i = 1:503
%         if y_train(1,i) > 0
%            y_train(1,i) = 1;
%         else if y_train(1,i) < 0
%                 y_train(1,i) = 0;
%             end
%         end   
%     end
%     e = d_train - y_train;
%     w = w + eta1 * e * X_train';
%     num_same_train = sum(y_train == d_train);
%     train_acc(n) = num_same_train/503;
%     
%     y_test = w * X_test;
%     for i = 1:167
%         if y_test(1,i) > 0
%            y_test(1,i) = 1;
%         else if y_test(1,i) < 0
%                 y_test(1,i) = 0;
%             end
%         end   
%     end
%     num_same_test =  sum(y_test == d_test);
%     test_acc(n) = num_same_test/167;
% end
% 
% plot(train_acc);
% hold on
% plot(test_acc);
% legend('train accuracy','test accuracy');
% title('256*256');

%% MLP of batch mode with overfitting analysis
% t = zeros(2,503);
% train_class = zeros(1,503);
% for i = 1:503
%     if train_label(i)==0
%         t(1,i) = 1;
%         train_class(i) = 1;
%     else if train_label(i)==1
%             t(2,i) = 1;
%             train_class(i) = 2;
%         end
%     end
% end
% 
% test_class = zeros(1,167);
% for i = 1:167
%     if test_label(i)==0
%         test_class(i) = 1;
%     else if test_label(i)==1
%             test_class(i) = 2;
%         end
%     end
% end
% 
% net = patternnet(10);
% net.trainParam.epochs = 500;
% net.trainParam.max_fail = 100;
% net.trainParam.goal = 0.000001;
% net.performParam.regularization = 0.3;
% 
% net = train(net,train_img,t);
% y_train = net(train_img);
% classes_train = vec2ind(y_train);
% y_test = net(test_img);
% classes_test = vec2ind(y_test);
% 
% same_train = sum(classes_train == train_class);
% train_acc = same_train/503;
% 
% same_test = sum(classes_test == test_class);
% test_acc = same_test/167;


%% MLP with sequential mode
nhidden = 10;
epochs = 300;
[net, acc_train] = train_seq(nhidden, train_img,train_label,train_num,epochs);
y_test = net(test_img);
acc_test = 1 - mean(abs(y_test-test_label));










