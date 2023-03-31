clc
clear
set(0,'defaultfigurecolor','w')
%% initialize
t = -0.5:0.01:1.5;
[x,y] = meshgrid(t);
z = (1-x).^2 + 100.*(y-x.^2).^2;
x0 = 0;
y0 = 0.5;

P = zeros(2,1000);
P(1,1) = x0;
P(2,1) = y0;
Z = zeros(1,1000);

time = 1;

%% iteration
for i=2:1000
    [fx,fy,Z(1,i-1)] = gradient_vector(P(1,i-1),P(2,i-1));
    g = [fx;fy];
    
    if (Z(1,i-1) > 0.0001) 
        time = time + 1;
    end
    
    H = Hessian(P(1,i-1),P(2,i-1));
    P(:,i) = P(:,i-1) - inv(H)*g;
    
end
info = ['There are ',int2str(time),' iterations for convergence.'];
disp(info);

%% 3 dimension plot
subplot(1,2,1)
mesh(x,y,z)
xlabel('x');
ylabel('y');
hold on

plot3(P(1,:),P(2,:),Z,'r','linewidth',1);
plot3(P(1,:),P(2,:),Z,'k.','markersize',8);

%% 2 dimension plot
subplot(1,2,2)
contour(x,y,z,30);
hold on

plot3(P(1,:),P(2,:),Z,'r','linewidth',1);
plot3(P(1,:),P(2,:),Z,'k.','markersize',8);


