clc;

theta1 = pi/4;
theta2 = pi/3;
theta3 = pi/6;

L1 = 0.055;
L2 = 0.315;
L3 = 0.045;
L4 = 0.108;
L5 = 0.005;
L6 = 0.034;
L7 = 0.015;
L8 = 0.088;
L9 = 0.204;

x3 = 0;
y3 = 0;
z3 = 0;

x2 = x3 * cos(theta3) - (y3-L9)*sin(theta3);
y2 = x3 * sin(theta3) + (y3-L9)*cos(theta3);
z2 = z3;

x1 = (x2+L7)*cos(theta2) - (y2-L8)*sin(theta2);
y1 = (x2+L7)*sin(theta2) + (y2-L8)*cos(theta2);
z1 = z2 - L5;

x0 = (x1+L6)*cos(theta1) + (z1+L4)*sin(theta1)
y0 = (x1+L6)*sin(theta1) - (z1+L4)*cos(theta1)
z0 = y1 + L2 + L3