
function [out]=AVOA(problem, params)

    
    % Problem definition
    fobj = problem.fobj;
    variables_no = problem.variables_no;
    lower_bound = problem.lower_bound;
    upper_bound = problem.upper_bound;
    DesiredPose = problem.DesiredPose;
    UnitChosen = 1;
    UnitApplied = 1000;
    Dimension = problem.Dimension;
    ObjectiveType = problem.ObjectiveType;
    robot = problem.Robot;
    DesiredTolerance = problem.DesiredTolerance;

    % Parameters choice
    pop_size = params.pop_size;
    max_iter = params.max_iter;
    W = params.Weights;
    ShowIterInfo = params.ShowIterInfo;

    % initialize Best_vulture1, Best_vulture2
    Best_vulture1_X=zeros(1,variables_no);
    Best_vulture1_F=inf;
    Best_vulture2_X=zeros(1,variables_no);
    Best_vulture2_F=inf;

    %Initialize the first random population of vultures
    X=initialization(pop_size,variables_no,upper_bound,lower_bound);

    %%  Controlling parameter
    p1=0.6;
    p2=0.4;
    p3=0.6;
    alpha=0.8;
    betha=0.2;
    gamma=2.5;

    %%Main loop
    current_iter=0; % Loop counter

    while current_iter < max_iter
        for i=1:size(X,1)
            % Calculate the fitness of the population
            current_vulture_X = X(i,:);
            %current_vulture_F=fobj(current_vulture_X);
            current_vulture_F=fobj(current_vulture_X, DesiredPose, UnitChosen, Dimension, ObjectiveType, W, robot);

            % Update the first best two vultures if needed
            if current_vulture_F<Best_vulture1_F
                Best_vulture1_F=current_vulture_F; % Update the first best bulture
                Best_vulture1_X=current_vulture_X;
            end
            if current_vulture_F>Best_vulture1_F && current_vulture_F<Best_vulture2_F
                Best_vulture2_F=current_vulture_F; % Update the second best bulture
                Best_vulture2_X=current_vulture_X;
            end
        end

        a=unifrnd(-2,2,1,1)*((sin((pi/2)*(current_iter/max_iter))^gamma)+cos((pi/2)*(current_iter/max_iter))-1);
        P1=(2*rand+1)*(1-(current_iter/max_iter))+a;

        % Update the location
        for i=1:size(X,1)
            current_vulture_X = X(i,:);  % pick the current vulture back to the population
            F=P1*(2*rand()-1);  

            random_vulture_X=random_select(Best_vulture1_X,Best_vulture2_X,alpha,betha);
            
            if abs(F) >= 1 % Exploration:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound);
            elseif abs(F) < 1 % Exploitation:
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound);
            end

            X(i,:) = current_vulture_X; % place the current vulture back into the population
        end

        current_iter=current_iter+1;
        convergence_curve(current_iter)=Best_vulture1_F;

        X = boundaryCheck(X, lower_bound, upper_bound);

        fprintf("===> In Iteration [%d], best estimation of the global optimum is %4.4f \n ", current_iter,Best_vulture1_F );
        
        
        %% Stopping criteria
        [CurrentPose, PoseError] = monitorPoseError(Best_vulture1_X, DesiredPose, ObjectiveType, Dimension, robot, UnitChosen, UnitApplied, ShowIterInfo);
        
        if ObjectiveType == "position"
            if (UnitApplied*sqrt(sum((DesiredPose(1:3)-CurrentPose(1:3)).^2))) <= UnitApplied*DesiredTolerance(1) 
                disp(['Solution found within ' num2str(current_iter) ' iterations: Best Cost = ' num2str(Best_vulture1_F)]);
                fprintf("Desired Pose Vector:")
                DesiredPose  %#ok<NOPRT>                        
                fprintf("Estimated Pose Vector:")
                CurrentPose  %#ok<NOPRT>                                  
                fprintf("Estimated Joint Vector:")
                Best_vulture1_X  %#ok<NOPRT>
                fprintf("Position Error Joint Vector (mm):")              
                position_error = UnitApplied*sqrt(sum((DesiredPose(1:3)-CurrentPose(1:3)).^2)) %#ok<NOPRT>  

                solutionTaggedAs = 1;

                break                
            end
        elseif ObjectiveType == "orientation"
            if (PoseError(4)+PoseError(5)+PoseError(6))/3 <= rad2deg(DesiredTolerance(2))
                
                disp(['Solution found within ' num2str(current_iter) ' iterations: Best Cost = ' num2str(Best_vulture1_F)]);
                fprintf("Desired Pose Vector:")
                DesiredPose  %#ok<NOPRT>                        
                fprintf("Estimated Pose Vector:")
                CurrentPose  %#ok<NOPRT>                                  
                fprintf("Estimated Joint Vector:")
                Best_vulture1_X  %#ok<NOPRT>                                  
                fprintf("Position Error Joint Vector (mm):")              
                position_error = UnitApplied*sqrt(sum((DesiredPose(1:3)-CurrentPose(1:3)).^2)) %#ok<NOPRT>  
                orientation_error = rad2deg(sum(abs(DesiredPose(4:6)-CurrentPose(4:6)))/3) %#ok<NOPRT>

                solutionTaggedAs = 1;

                break
            end
        else
            if abs(DesiredPose(1)-CurrentPose(1)) < DesiredTolerance(1) && abs(DesiredPose(2)-CurrentPose(2)) < DesiredTolerance(1) && abs(DesiredPose(3)-CurrentPose(3)) < DesiredTolerance(1) && abs(DesiredPose(4)-CurrentPose(4)) < DesiredTolerance(2) && abs(DesiredPose(5)-CurrentPose(5)) < DesiredTolerance(2) && abs(DesiredPose(6)-CurrentPose(6)) < DesiredTolerance(2)

                disp(['Solution found within ' num2str(current_iter) ' iterations: Best Cost = ' num2str(Best_vulture1_F)]);
                fprintf("Desired Pose Vector:")
                DesiredPose  

ok<NOPRT>                        
                fprintf("Estimated Pose Vector:")
                CurrentPose  %#ok<NOPRT>     
                fprintf("Pose Error Joint Vector (mm/degree):") 
                PoseError = [UnitApplied*abs(DesiredPose(1)-CurrentPose(1)),  UnitApplied*abs(DesiredPose(2)-CurrentPose(2)),  UnitApplied*abs(DesiredPose(3)-CurrentPose(3)), ...
                             rad2deg(abs(DesiredPose(4)-CurrentPose(4))), rad2deg(abs(DesiredPose(5)-CurrentPose(5))), rad2deg(abs(DesiredPose(6)-CurrentPose(6)))]; 
                PoseError          %#ok<NOPRT>
                fprintf("Estimated Joint Vector:")
                Best_vulture1_X  %#ok<NOPRT>                                   
                %fprintf("Position Error Joint Vector (mm):")              
                position_error = UnitApplied*sqrt(sum((DesiredPose(1:3)-CurrentPose(1:3)).^2)) %#ok<NOPRT>  
                %fprintf("Orientation Error Joint Vector (degree):")         
                orientation_error = rad2deg(sum(abs(DesiredPose(4:6)-CurrentPose(4:6)))/3) %#ok<NOPRT>


                solutionTaggedAs = 1;

                break
            end
        end            
                
    end
    
    if current_iter >= max_iter 

        DH = getDH_rad_cec_vectorized(robot, Best_vulture1_X, UnitChosen);
        T_eval_current = forwardKinematics_rad_cec_vectorized(DH); 
        CurrentPose = getPose_rad_cec_vectorized(T_eval_current, Dimension)';

        disp(['Maximum Number of iterations reached ' num2str(current_iter) ' iterations: Best Cost = ' num2str(Best_vulture1_F)]);
        fprintf("Desired Pose Vector:")
        DesiredPose  %#ok<NOPRT>                        
        fprintf("Estimated Pose Vector:")
        CurrentPose  %#ok<NOPRT>                                   
        fprintf("Estimated Joint Vector:")
        CurrentJoints = Best_vulture1_X;  
        CurrentJoints %#ok<NOPRT>
        fprintf("Position Error Joint Vector (mm):") 
        position_error = UnitApplied*sqrt(sum((DesiredPose(1:3)-CurrentPose(1:3)).^2));
        positionError = [UnitApplied*abs(DesiredPose(1)-CurrentPose(1)),  UnitApplied*abs(DesiredPose(2)-CurrentPose(2)),  UnitApplied*abs(DesiredPose(3)-CurrentPose(3))];
        positionError %#ok<NOPRT>
        if ObjectiveType ~= "position"
            fprintf("Orientation Error Joint Vector (degree):")         
            orientation_error = rad2deg(sum(abs(DesiredPose(4:6)-CurrentPose(4:6)))/3);
            orientationError = [rad2deg(abs(DesiredPose(4)-CurrentPose(4))), rad2deg(abs(DesiredPose(5)-CurrentPose(5))), rad2deg(abs(DesiredPose(6)-CurrentPose(6)))];
            orientationError %#ok<NOPRT>

        end


        solutionTaggedAs = 0;

    end

    % Results
    out.pop = X;
    out.BestSolution = Best_vulture1_X;
    out.BestCost = Best_vulture1_F;
    out.ConvergenceCurve = convergence_curve;
    out.Iterations = current_iter;
    out.SolutionTaggedAs = solutionTaggedAs;
    if ObjectiveType == "position"
        out.PositionError = position_error;
    else
        out.PositionError = position_error;
        out.OrientationError = orientation_error;
    end
    

end






