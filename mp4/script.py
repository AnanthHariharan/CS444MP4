import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Output directory for logs and predictions
output_dir = os.getenv('OUTPUT_DIR', 'runs/default_run')
os.makedirs(output_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(output_dir)

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Datasets and loaders
train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, criterion, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Log training loss to TensorBoard
    writer.add_scalar("Training Loss", running_loss / len(train_loader), epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation and saving test predictions
model.eval()
correct = 0
total = 0
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions and true labels
        predictions.extend(zip(predicted.cpu().numpy(), labels.cpu().numpy()))

# Calculate accuracy and log to TensorBoard
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
writer.add_scalar("Test Accuracy", accuracy)
writer.close()

# Save test predictions to file
predictions_file = os.path.join(output_dir, "test_predictions.txt")
with open(predictions_file, "w") as f:
    for pred, true_label in predictions:
        f.write(f"Predicted: {pred}, True Label: {true_label}\n")

print(f"Test predictions saved to {predictions_file}")
