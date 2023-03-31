function H = Hessian(x,y)
   H(1,1) = 1200*(x^2) - 400*y + 2;
   H(1,2) = -400*x;
   H(2,1) = -400*x;
   H(2,2) = 200;
end
