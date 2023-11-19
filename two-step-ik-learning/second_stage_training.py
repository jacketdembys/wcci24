from model import *
from data_loading import *


"""
    This module is for second step training
    that is the idea of training a network 
    to solve the inverse kinematics, optimized
    in the cartesian space, using a pre-trained 
    network on Forward Kinematics.

"""

robot_choice = "3DoF-3R"
mode_choice = "IKFK"
test_size = 0.2
batch_size = 32
IKFK_train_loader, IKFK_test_loader, pos_shape, joints_shape = data_loader(robot_choice, mode_choice, test_size, batch_size)


input_size = pos_shape
middle_state_size = joints_shape
s1_hidden_list = [32, 64, 128]
s2_hidden_list = [32, 64, 128, 64]
output_size = pos_shape
second_network_path = './model_weights/test_2.pth'  # Replace with actual path

# Initialize the network
my_network = IK_Network(input_size, s1_hidden_list, s2_hidden_list, middle_state_size, output_size, second_network_path)

# print("Done!")
learning_rate = 0.0001
num_epochs = 300
criterion = nn.MSELoss()
test_criterion = nn.L1Loss()
optimizer = optim.Adam(my_network.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in IKFK_train_loader:
        inputs = batch['data']
        labels = batch['targets'].squeeze()
        
        # Forward pass
        outputs = my_network(inputs)
        loss = test_criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
my_network.eval()
correct = 0
total = 0
mean_loss = []
with torch.no_grad():
    for batch in IKFK_test_loader:
        inputs = batch['data']
        labels = batch['targets'].squeeze()
        outputs = my_network(inputs)
#         _, predicted = torch.max(outputs.data, 1)
        loss = test_criterion(outputs, labels)
        mean_loss.append(loss.item())
#         correct += (predicted == labels).sum().item()
mean_loss = np.mean(np.array(mean_loss))
print(f'Test Error: {mean_loss}')






