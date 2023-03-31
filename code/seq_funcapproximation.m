function [ net, accu_train ] = seq_funcapproximation( n, images, labels,train_num, epochs )
% % 1. Change the input to cell array form for sequential training
% images_c = num2cell(images, 1);
% labels_c = num2cell(labels, 1);
% 
% % 2. Construct and configure the MLP
% net = fitnet(n);
% 
% net.divideFcn = 'dividetrain'; % input for training only
% net.trainFcn = 'traingdx'; % 'trainrp' 'traingdx'
% net.trainParam.epochs = epochs;
% 
% accu_train = zeros(epochs,1); % record accuracy on training set of each epoch
% 
% % 3. Train the network in sequential mode
%     for i = 1 : epochs
%         display(['Epoch: ', num2str(i)])
%         idx = randperm(train_num); % shuffle the input
%         net = train(net, images_c(idx), labels_c(idx));
%         pred_train = net(images(1:train_num)); % predictions on training set
%         accu_train(i) = 1 - mean(abs(pred_train-labels(1:train_num)));
%     end
images_c = num2cell(images, 1);
labels_c = num2cell(labels, 1);

% 2. Construct and configure the MLP
net = fitnet(n);

net.divideFcn = 'dividetrain'; % input for training only
net.trainFcn = 'traingdx'; % 'trainrp' 'traingdx'
net.trainParam.epochs = epochs;

accu_train = zeros(epochs,1); % record accuracy on training set of each epoch
% 3. Train the network in sequential mode
    for i = 1 : epochs

        display(['Epoch: ', num2str(i)])

        idx = randperm(train_num); % shuffle the input

        net = train(net, images_c(:,idx), labels_c(:,idx));

        pred_train = net(images(:,1:train_num)); % predictions on training set
        accu_train(i) = 1 - mean(abs(pred_train-labels(1:train_num)));

    end
end