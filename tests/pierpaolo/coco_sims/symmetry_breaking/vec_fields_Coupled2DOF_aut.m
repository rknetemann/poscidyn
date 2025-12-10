function y = vec_fields_Coupled2DOF_aut(x, p)
% Encoding of a non-autonomous vector field.

% x1 = x(1,:);
% x2 = x(2,:);
% 
% U3=x(3,:);
% U4=x(4,:);
% 
% p1 = p(1,:);
% p2 = p(2,:);
% p3 = p(3,:);
% p4 = p(4,:);
% 
% y(1,:) = x2;
% y(2,:) = -p3.*x2-(1+p2.*U3).*x1-p4.*x1.^3;
%     y(3,:)= U3+2*pi./p1.*U4-U3.*((U3.^2)+(U4.^2));
%     y(4,:)=-2*pi./p1.*U3+U4-U4.*((U3.^2)+(U4.^2));


% -------------------------------------------------------
% Vectorized right-hand side for 2DOF nonlinear system
% x = [x1; x2; x3; x4; U5; U6]
% p = [fx; fy; omega]
% -------------------------------------------------------

% -------------------------
% State variables
% -------------------------
x1 = x(1,:); % x
x2 = x(2,:); % x'
x3 = x(3,:); % y
x4 = x(4,:); % y'
u5 = x(5,:); % cos(ωt)
u6 = x(6,:); % sin(ωt)

% -------------------------
% Parameters
% -------------------------
fx  = p(1,:);   % drive amplitude on x
fy  = p(3,:);   % drive amplitude on y
omega = p(2,:);   % drive frequency

% -------------------------
% Fixed physical constants
% -------------------------
Qx = 10.0;
Qy = 20.0;

omega0x = 1.0;
omega0y = 2.0;

gamx = 2.67e-2;
gamy = 5.40e-1;

alpha = 3.74e-1;

% -------------------------
% Nonlinear terms
% -------------------------
z1 = gamx .* x1.^3; 
z2 = gamy .* x3.^3; 

% -------------------------
% Output initialization
% -------------------------
y = x;
 
y(1,:) = x2; 
y(2,:) = -(omega0x/Qx).*x2  - omega0x^2 .* x1  - z1  - 2*alpha .* x1 .* x3  + fx .* u5; 
y(3,:) = x4; 
y(4,:) = -(omega0y/Qy).*x4  - omega0y^2 .* x3  - z2 - alpha .* (x1.^2)  + fy .* u5;
 
y(5,:) = u5 + omega.*u6 - u5 .* (u5.^2 + u6.^2); 
y(6,:) = -omega.*u5 + u6  - u6 .* (u5.^2 + u6.^2);

end


