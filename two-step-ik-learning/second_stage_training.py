from model import *
from data_loading import *
from utils import *
import sys


"""
    This module is for second step training
    that is the idea of training a network 
    to solve the inverse kinematics, optimized
    in the cartesian space, using a pre-trained 
    network on Forward Kinematics.

"""

def custom_loss(pose_pred, pose_true, middle_state, robot_choice, alpha=0.5, beta=0.5):
    mse_loss = F.mse_loss(pose_pred, pose_true)  # Compute the mean squared error
    # print(mse_loss)
    rec_pose = reconstruct_pose(middle_state, robot_choice)
    # print(rec_pose)
    reg_mse_loss = F.mse_loss(rec_pose, pose_true)  # L2 regularization term
    # print(reg_mse_loss)
    total_loss = alpha * mse_loss + beta * reg_mse_loss  # Combine with a regularization term
    # print(total_loss)
    # sys.exit()
    return total_loss


robot_choice = "3DoF-3R"
mode_choice = "IKFK"
test_size = 0.2
batch_size = 32
IKFK_train_loader, IKFK_test_loader, pos_shape, joints_shape = data_loader(robot_choice, mode_choice, test_size, batch_size)


input_size = pos_shape
middle_state_size = joints_shape
s1_hidden_list = [64, 64, 64]
s2_hidden_list = [64, 128]
second_model_choice = "MLP"
output_size = pos_shape
second_network_path = './model_weights/MLP_withoutJ_1.pth'  # Replace with actual path

# Initialize the network
my_network = IK_Network(input_size, s1_hidden_list, s2_hidden_list, middle_state_size, output_size,second_model_choice, second_network_path)

# print("Done!")
learning_rate = 0.0001
num_epochs = 100
# criterion = nn.MSELoss()
test_criterion = nn.L1Loss()
optimizer = optim.Adam(my_network.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in IKFK_train_loader:
        inputs = batch['data']
        labels = batch['targets'].squeeze()
        
        # Forward pass
        outputs, middle_state = my_network(inputs)
        loss = custom_loss(outputs, labels, middle_state, robot_choice, alpha=1, beta=0.5)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
my_network.eval()
correct = 0
total = 0
mean_loss_dh = []
mean_loss_n = []
with torch.no_grad():
    for batch in IKFK_test_loader:
        inputs = batch['data']
        labels = batch['targets'].squeeze()
        outputs, middle_state = my_network(inputs)
#         _, predicted = torch.max(outputs.data, 1)
        rec_pose = reconstruct_pose(middle_state, robot_choice)
        # print(labels.shape)
        # print(rec_pose.shape)
        loss_dh = test_criterion(rec_pose, labels)
        loss_n = test_criterion(outputs, labels)
        mean_loss_dh.append(loss_dh.item())
        mean_loss_n.append(loss_n.item())
#         correct += (predicted == labels).sum().item()
mean_loss_n = np.mean(np.array(mean_loss_n))
mean_loss_dh = np.mean(np.array(mean_loss_dh))
print(f'DH Test Error: {mean_loss_dh}')
print(f'Network Test Error: {mean_loss_n}')






