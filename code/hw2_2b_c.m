clc
clear
set(0,'defaultfigurecolor','w')
%% sampling points in the domain of [-1,1]
x=-1:0.05:1;
x_true = -3:0.01:3;
%% generating training data, and the desired outputs
y=1.2*sin(pi.*x)-cos(2.4.*pi.*x);
y_true = 1.2*sin(pi.*x_true)-cos(2.4.*pi.*x_true);
%% specify the structure and learning algorithm for MLP
n_hidden = 1;
net = feedforwardnet(n_hidden,'trainbr');
%% Train the MLP
[net,tr]=train(net,x,y);
%% Test the MLP, net_output is the output of the MLP, ytest is the desired output.
xtest=-1:0.01:1;
net_test = sim(net,xtest);
ytest=1.2*sin(pi.*xtest)-cos(2.4.*pi.*xtest);
net_x = -3:0.01:3;
net_output=sim(net,net_x);
n = length(xtest);
mes_test = (1/n) * (sum((net_test-ytest).^2))
%% Plot out the test results
plot(xtest,ytest,'k.','markersize',5);
hold on;
plot(net_x,net_output,'r','linewidth',1);
hold on;
plot(x_true,y_true,'--','color',[244 168 47]/255);
axis([-3 3 -3 3]);
legend('test samples','NN outputs','true function')
title('hidden layer size = 1');
