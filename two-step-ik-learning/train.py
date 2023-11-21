from model import *
from data_loading import *

"""
    This module trains the first step of the idea
    which is training a network on Forward Kinematics
    and using that later in the second stage.

"""

robot_choice = "3DoF-3R"
mode_choice = "FK"
test_size = 0.2
batch_size = 32
FK_train_loader, FK_test_loader, input_size, output_size = data_loader(robot_choice, mode_choice, test_size, batch_size)


input_size = input_size
hidden_sizes = [64, 64, 64, 64]
output_size = output_size
learning_rate = 0.0001
num_epochs = 300

model_choice = "ResMLP"

if model_choice == "MLP":
    model = MLP(input_size, hidden_sizes, output_size)
elif model_choice == "ResMLP":
    model = ResMLP(input_size, hidden_sizes, output_size)

criterion = nn.MSELoss()
test_criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Training loop
for epoch in range(num_epochs):
    for batch in FK_train_loader:
        inputs = batch['data']
        labels = batch['targets'].squeeze()
        
        # Forward pass
        outputs = model(inputs)
        loss = test_criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
correct = 0
total = 0
mean_loss = []
with torch.no_grad():
    for batch in FK_test_loader:
        inputs = batch['data']
        labels = batch['targets'].squeeze()
        outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
        loss = test_criterion(outputs, labels)
        mean_loss.append(loss.item())
#         correct += (predicted == labels).sum().item()
mean_loss = np.mean(np.array(mean_loss))
print(f'Test Error: {mean_loss}')

model_weights_path = './model_weights/Res_MLP_Test_1.pth'
# Save the model's state_dict to the specified path
torch.save(model.state_dict(), model_weights_path)