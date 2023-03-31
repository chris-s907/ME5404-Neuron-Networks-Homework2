clc
clear
set(0,'defaultfigurecolor','w')
%% initialize
t = -0.5:0.01:1.5;
[x,y] = meshgrid(t);
z = (1-x).^2 + 100.*(y-x.^2).^2;
x0 = 0;
y0 = 0.5;
% eta = 0.001;
eta = 0.002;

X = zeros(1,100000);
X(1,1) = x0;
Y = zeros(1,100000);
Y(1,1) = y0;
Z = zeros(1,100000);
time = 1;

%% iteration
for i=2:100000
    [fx,fy,Z(1,i-1)] = gradient_vector(X(1,i-1),Y(1,i-1));
    
    if (Z(1,i-1) > 0.0001) 
        time = time + 1;
    end
    
    X(1,i) = X(1,i-1) - eta*fx;
    Y(1,i) = Y(1,i-1) - eta*fy;
end
info = ['There are ',int2str(time),' iterations for convergence.'];
disp(info);

%% 3 dimension plot
subplot(1,2,1)
mesh(x,y,z)
xlabel('x');
ylabel('y');
hold on

plot3(X,Y,Z,'r','linewidth',1);
plot3(X,Y,Z,'k.','markersize',8);

%% 2 dimension plot
subplot(1,2,2)
contour(x,y,z,30);
hold on

plot3(X,Y,Z,'r','linewidth',1);
plot3(X,Y,Z,'k.','markersize',8);


