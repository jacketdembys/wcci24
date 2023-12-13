
L(1) = Link('revolute', 'd', 0.0,       'a', 0.0,       'alpha', -pi/2);
L(2) = Link('revolute', 'd', 0.14909,   'a', 0.4318,    'alpha', 0.0);
L(3) = Link('revolute', 'd', 0.0,       'a', -0.02032,  'alpha', pi/2);
L(4) = Link('revolute', 'd', 0.43307,   'a', 0.0,       'alpha', -pi/2);
L(5) = Link('revolute', 'd', 0.0,       'a', 0.0,       'alpha', pi/2);
L(6) = Link('revolute', 'd', 0.05625,   'a', 0.0,       'alpha', 0.0);
robot = SerialLink(L);
robot.name = "6DoF-6R-Puma560";



rng(2023)
samples = 1000;
data = zeros(samples, 12);

for i=1:samples
    Q_min = [-360;-360;-360;-360;-360;-360];
    Q_max = [360;360;360;360;360;360];

    Q = unifrnd(Q_min,Q_max,[6,1]);
    T = robot.fkine(Q);
    D = T.t;
    O = T.rpy();
end

Q
T