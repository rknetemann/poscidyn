delta = 0.000001;
F = inline('y^2 - y^3','t','y');
opts = odeset('RelTol',1.e-4);
ode23s(F,[0 2/delta],delta,opts);