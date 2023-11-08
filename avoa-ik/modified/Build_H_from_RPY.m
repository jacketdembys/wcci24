function T = Build_H_from_RPY(D)

    %{
    X = D(1);
    Y = D(2); 
    Z = D(3); 
    r = D(4); 
    p = D(5); 
    y = D(6);
    
    
    T = [cos(r)*cos(p), cos(r)*sin(p)*sin(y) - sin(r)*cos(y), cos(r)*sin(p)*cos(y) + sin(r)*sin(y), X;
         sin(r)*cos(p), sin(r)*sin(p)*sin(y) + cos(r)*cos(y), sin(r)*sin(p)*cos(y) - cos(r)*sin(y), Y;
               -sin(p),                        cos(p)*sin(y),                        cos(p)*cos(y), Z;
                     0,                                    0,                                    0, 1];
    %}
    
    
    rot = eul2rotm(D(4:6), 'XYZ');
    T = eye(4);
    T(1:3, 1:3) = rot;
    T(1:3, 4) = D(1:3);    
    
end