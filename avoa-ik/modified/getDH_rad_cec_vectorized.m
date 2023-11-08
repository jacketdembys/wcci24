% Retrieve the DH table of a robotic manipulator.
% Example: DH = getDH(robot, Q_initial)
% Inputs:  robot = a string representing the robot to load
%          Q_initial = a vector representing the initial joint
%          configuration of the robot to load
% Outputs: DH = a matrix representing the corresponding DH table

function DH = getDH_rad_cec_vectorized(robot, Q_initial, U)

    % this is the Puma to experiment the Decoupling strategy of the
    % position and orientation joints
    if (strcmp(robot, 'RRRRRR2')) 

    DH = [Q_initial(1),             0.0,        0.0,       -pi/2;
          Q_initial(2),         0.14909,     0.4318,         0.0;
          Q_initial(3),             0.0,   -0.02032,        pi/2;
          Q_initial(4),         0.43307,        0.0,       -pi/2;
          Q_initial(5),             0.0,        0.0,        pi/2;
          Q_initial(6),         0.05625,        0.0,         0.0];

    % convert the entries of the DH table
    DH(:,2) = U*DH(:,2);
    DH(:,3) = U*DH(:,3); 

    % vectorized DH
    [nPop,~] = size(Q_initial);
    DH = zeros(6,4,nPop);
    DH(:,1,:) = Q_initial';
    DH(:,2,:) = repmat([0.0; 0.14909;      0.0;  0.43307;   0.0; 0.05625], 1, nPop);
    DH(:,3,:) = repmat([0.0;  0.4318; -0.02032;      0.0;   0.0;     0.0], 1, nPop);
    DH(:,4,:) = repmat([-pi/2; 0.0; pi/2; -pi/2; pi/2; 0.0], 1, nPop);
        
      
    %% this is the kinova
    elseif (strcmp(robot, 'RRRRRR') || strcmp(robot, 'kinova6'))   
        
        D1 = 0.2755;
        D2 = 0.4100;
        D3 = 0.2073;
        D4 = 0.0741;
        D5 = 0.0741;
        D6 = 0.1600;
        e2 = 0.0098;
        aa = ((30.0*pi)/180.0);
        ca = cos(aa);
        sa = sin(aa);
        c2a = cos(2*aa);
        s2a = sin(2*aa);
        d4b = (D3 + sa/s2a *D4);
        d5b = (sa/s2a*D4 + sa/s2a *D5);
        d6b = (sa/s2a*D5 + D6);
        
        DH = [-Q_initial(1),             D1,          0.0,      pi/2;
               Q_initial(2)+(pi/2),      0.0,         D2,       pi;
               Q_initial(3)-(pi/2),     -e2,          0.0,      pi/2;
               Q_initial(4),            -d4b,         0.0,      2*aa;
               Q_initial(5)+(pi),       -d5b,         0.0,      2*aa;
               Q_initial(6)-(pi/2),     -d6b,         0.0,      pi];  
           
        % convert the entries of the DH table
        DH(:,2) = U*DH(:,2);
        DH(:,3) = U*DH(:,3);
        
        % vectorized DH
        [nPop,~] = size(Q_initial);
        Q_initial_used = Q_initial;
        Q_initial_used(:,1) = Q_initial(:,1);
        Q_initial_used(:,2) = Q_initial(:,2)+(pi/2);
        Q_initial_used(:,3) = Q_initial(:,3)-(pi/2);
        Q_initial_used(:,4) = Q_initial(:,4);
        Q_initial_used(:,5) = Q_initial(:,5)+(pi);
        Q_initial_used(:,6) = Q_initial(:,6)-(pi/2);        
        
        DH = zeros(6,4,nPop);
        DH(:,1,:) = Q_initial_used';
        DH(:,2,:) = repmat([  D1; 0.0;  -e2; -d4b; -d5b; -d6b], 1, nPop);
        DH(:,3,:) = repmat([ 0.0;  D2;  0.0;  0.0;  0.0;  0.0], 1, nPop);
        DH(:,4,:) = repmat([pi/2;  pi; pi/2; 2*aa; 2*aa;   pi], 1, nPop);
        
        
    elseif (strcmp(robot, 'RRRRRRR')) 
        
        DH = [Q_initial(1),             0.0,        0.0,        -pi/2;
              Q_initial(2),             0.0,        0.0,         pi/2;
              Q_initial(3),             0.55,       0.045,      -pi/2;
              Q_initial(4),             0.0,       -0.045,       pi/2;
              Q_initial(5),             0.3,        0.0,        -pi/2;
              Q_initial(6),             0.0,        0.0,         pi/2;
              Q_initial(7),             0.06,       0.0,         0];
        
        % convert the entries of the DH table
        DH(:,2) = U*DH(:,2);
        DH(:,3) = U*DH(:,3); 
        
        % vectorized DH
        [nPop,~] = size(Q_initial);
        DH = zeros(7,4,nPop);
        DH(:,1,:) = Q_initial';
        DH(:,2,:) = repmat([0.0; 0.0; 0.55; 0.0; 0.3; 0.0; 0.06], 1, nPop);
        DH(:,3,:) = repmat([0.0; 0.0; 0.045; -0.045; 0.0; 0.0; 0.0], 1, nPop);
        DH(:,4,:) = repmat([-pi/2; pi/2; -pi/2; pi/2; -pi/2; pi/2; 0.0], 1, nPop);
        
        
       
    end     

end