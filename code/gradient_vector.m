function [dx,dy,z] = gradient_vector(x,y)
    dx = 2.*(x-1) - 400.*x.*(y-x.^2);
    dy = 200.*(y-x.^2);
    z = (1-x).^2 + 100.*(y-x.^2).^2;
end
