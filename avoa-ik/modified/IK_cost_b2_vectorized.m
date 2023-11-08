function obj = IK_cost_b2_vectorized(Q, D_desired, U, d, ot, w, robot)

    %robot = 'RRRRRR';
    DH = getDH_rad_cec_vectorized(robot, Q, U);
    T = forwardKinematics_rad_cec_vectorized(DH); 
    D_current = getPose_rad_cec_vectorized(T, d)';  
        
    if d == 3

        % Objective function
        if ot == "position"
            obj = sqrt(sum((D_desired(1:3)-D_current(:,1:3)).^2, 2));  
        end
        
    elseif d == 6 

        % RPY based Objective function
        if ot == "poserpyss"  
            
            [nPop, ~] = size(Q);
            
            obj_p = sqrt(sum((D_desired(1:3)-D_current(:,1:3)).^2, 2));             
            
            T_current = T;
            T_desired = Build_H_from_RPY(D_desired); 
            R_temp = reshape(repmat(T_desired(1:3,1:3), 1,nPop), 3,3,1,nPop);
            
            % TODO: I am unable to avoid a for loop here, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            R_error = zeros(3,3,nPop);
            for i=1:nPop
                R_error(:,:,i) = R_temp(:,:,1,i)/T_current(1:3,1:3,:,i);
            end            
            
            quat_error = rotm2quat(R_error);
            %obj = norm(eye(3)) - norm(quat_error);
            obj_o = quat2axang(quat_error);
            obj_o = obj_o(:,4);
           
                       
            %w1 = 0.8;
            %w2 = 0.2;
            
            %w1 = w(1);
            %w2 = w(2);
            
            w1 = w(1)/sum(w);
            w2 = w(2)/sum(w);
            
            obj = w1.*obj_p + w2.*obj_o;  
            
            %obj_p_2 = obj_p./(obj_p+obj_o);
            %obj_o_2 = obj_o./(obj_p+obj_o);
            %obj = w1.*obj_p_2 + w2.*obj_o_2;
            
        % RPY average based objective
        elseif ot == "poserpys"
            
            obj_p = sqrt(sum((D_desired(1:3)-D_current(:,1:3)).^2, 2));               
            obj_o = sum(abs(D_desired(4:6)-D_current(:,4:6)),2)./3; 
            w1 = w(1)/sum(w);
            w2 = w(2)/sum(w);
            
            obj = w1.*obj_p + w2.*obj_o; 
            
        % Rotation based objective
        elseif ot == "poserotation"
            
            [nPop, ~] = size(Q);
            
            obj_p = sqrt(sum((D_desired(1:3)-D_current(:,1:3)).^2, 2));             
            
            T_current = T;
            T_desired = Build_H_from_RPY(D_desired); 
            R_temp = reshape(repmat(T_desired(1:3,1:3), 1,nPop), 3,3,1,nPop);            
            R_error = zeros(3,3,nPop);
            obj_o = zeros(nPop, 1);
            for i=1:nPop
                %R_error(:,:,i) = R_temp(:,:,1,i)/T_current(1:3,1:3,:,i);
                R_error(:,:,i) = R_temp(:,:,1,i)*T_current(1:3,1:3,:,i)';
                obj_o(i) = acos((trace(R_error(:,:,i))-1)/2);
            end              
            
            w1 = w(1)/sum(w);
            w2 = w(2)/sum(w);
            
            obj = w1.*obj_p + w2.*obj_o;      
            
        elseif ot == "posedoublequaternion"   
            
            [nPop, ~] = size(Q);
            T_current = T;
            T_desired = Build_H_from_RPY(D_desired); 
            T_desired = reshape(repmat(T_desired(1:4,1:4), 1,nPop), 4,4,1,nPop);
            
            dq_current = getDualQuat(reshape(T_current(:,:,1,:),4,4,nPop));
            %dq_desired = getDualQuat(T_desired);
            dq_desired = getDualQuat(reshape(T_desired(:,:,1,:),4,4,nPop));
            
            %dq_error_desired = DQmult(dq_desired, DQconj(dq_desired, 'line'));
            dq_error_current = DQmult(dq_desired, DQconj(dq_current, 'line')); 
            
            %{
            % set the orientation quaternion
            dq_error_rotation = zeros(8,nPop);
            dq_error_rotation(1:4,:) = dq_error_current(1:4,:);
            
            % set the psotion quaternion
            dq_error_position = zeros(8,nPop);
            dq_error_position(5:end,:) = dq_error_current(5:end,:);            
            dq_error_position(1,:) = ones(1,nPop);   
            
            [px, pd] = dquat2trans(dq_error_position);
            [od, ox] = dquat2rot(dq_error_rotation); 
            %}
            
            [px, pd] = dquat2trans(dq_error_current);
            [od, ox] = dquat2rot(dq_error_current);
            obj_p = pd;
            obj_o = deg2rad(od);
            %obj = abs(norm(dq_error_desired)-norm(dq_error_current));
            
            w1 = w(1)/sum(w);
            w2 = w(2)/sum(w);
            
            obj = w1.*obj_p + w2.*obj_o;   
            
            %{
            dq_error = DQmult(dq_desired, dq_current);            
            dq_error_c = DQconj(dq_error);             
            obj = norm(DQmult(dq_error,dq_error_c))';
            %}
            
        
        elseif ot == "posequaternion"
            
            [nPop, ~] = size(Q);
            
            obj_p = sqrt(sum((D_desired(1:3)-D_current(:,1:3)).^2, 2));             
            
            R_current = reshape(T(1:3,1:3,1,:),[3,3,nPop]);
            T_desired = Build_H_from_RPY(D_desired); 
            R_desired = reshape(repmat(T_desired(1:3,1:3), 1,nPop), 3,3,nPop);            
            q_current = rotm2quat(R_current);
            q_desired = rotm2quat(R_desired);
            %q_current = normalize(quaternion(q_current));
            %q_desired = normalize(quaternion(q_desired));
            %q_current_conj = conj(q_current);
            %q_error = q_desired.*q_current_conj;
            %q_desired_r = compact(q_desired);
            %q_current_conj_r = compact(q_current_conj);
            %obj_o = 2*acos(sqrt(sum((q_current.*q_desired).^2, 2)));                     
            obj_o = acos(min(1,2*(sum(q_current(:,:).*q_desired(:,:),2).^2)-1));
            
            w1 = w(1)/sum(w);
            w2 = w(2)/sum(w);
            
            obj = w1.*obj_p + w2.*obj_o;  
            
        elseif ot == "orientation"
            [nPop, ~] = size(Q);
            T_current = T;
            T_desired = Build_H_from_RPY(D_desired); 
            R_temp = reshape(repmat(T_desired(1:3,1:3), 1,nPop), 3,3,1,nPop);
            
            % TODO: I am unable to avoid a for loop here, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            R_error = zeros(3,3,nPop);
            for i=1:nPop
                R_error(:,:,i) = R_temp(:,:,1,i)/T_current(1:3,1:3,:,i);
            end            
            
            quat_error = rotm2quat(R_error);
            %obj = norm(eye(3)) - norm(quat_error);
            obj_o = quat2axang(quat_error);
            obj = obj_o(:,4);            
                    
        end
    end

end