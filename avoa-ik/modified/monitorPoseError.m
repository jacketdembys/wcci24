function [CurrentPose, PoseError] = monitorPoseError(Best_vulture1_X, DesiredPose, ObjectiveType, Dimension, robot, UnitChosen, UnitApplied, ShowIterInfo)


    %% From best vulture 1     
    DH = getDH_rad_cec_vectorized(robot, Best_vulture1_X, UnitChosen);
    T_eval_current = forwardKinematics_rad_cec_vectorized(DH);
    CurrentPose = getPose_rad_cec_vectorized(T_eval_current, Dimension)';

    if ObjectiveType == "position"
        PoseError = [UnitApplied*abs(DesiredPose(1)-CurrentPose(1)),  UnitApplied*abs(DesiredPose(2)-CurrentPose(2)),  UnitApplied*abs(DesiredPose(3)-CurrentPose(3))];  
    %elseif ObjectiveType == "orientation"
    %    PoseError = [rad2deg(abs(DesiredPose(4)-CurrentPose(4))), rad2deg(abs(DesiredPose(5)-CurrentPose(5))), rad2deg(abs(DesiredPose(6)-CurrentPose(6)))];  
    else
        PoseError = [UnitApplied*abs(DesiredPose(1)-CurrentPose(1)),  UnitApplied*abs(DesiredPose(2)-CurrentPose(2)),  UnitApplied*abs(DesiredPose(3)-CurrentPose(3)), ...
                     rad2deg(abs(DesiredPose(4)-CurrentPose(4))), rad2deg(abs(DesiredPose(5)-CurrentPose(5))), rad2deg(abs(DesiredPose(6)-CurrentPose(6)))];  
    end

    if ShowIterInfo    
        fprintf("\n")
        fprintf("Desired Pose Vector:")
        DesiredPose  %#ok<NOPRT>                        
        fprintf("Estimated Pose Vector:")
        CurrentPose  %#ok<NOPRT>     
        fprintf("Pose Error Joint Vector (mm/degree):") 
        PoseError %#ok<NOPRT>
        fprintf("Estimated Joint Vector:")
        Best_vulture1_X  %#ok<NOPRT>
        fprintf("\n")
    end

end