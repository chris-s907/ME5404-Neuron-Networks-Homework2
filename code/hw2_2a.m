clc 
clear
set(0,'defaultfigurecolor','w')
%% Generate training and testing data
train_x = -1:0.05:1;
train_y = 1.2*sin(pi*train_x) - cos(2.4*pi*train_x);
test_x = -1:0.01:1;
test_y = 1.2*sin(pi*test_x) - cos(2.4*pi*test_x);

%% parameter setting
n = 50;
epochs = 100;
train_num = size(train_x,2);
[net, acc_train] = seq_funcapproximation(n,train_x,train_y,train_num,epochs);

x = -3:0.01:3;
y = 1.2*sin(pi*x) - cos(2.4*pi*x);
net_output=sim(net,x);
net_y = sim(net,test_x);
n_test = length(test_x);
mes_test = (1/n_test) * (sum((net_y-test_y).^2))
%% figure
plot(test_x,test_y,'k.','markersize',5);
hold on;

plot(x,net_output,'r','linewidth',1);
hold on;

plot(x,y,'--','color',[244 168 47]/255);
hold on;
axis([-3 3 -3 3]);
legend('test samples','NN outputs','true function')
title('hidden layer size = 50');