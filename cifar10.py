import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf

# 1. DATASET + DATALOADER
transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 2. CNN MODEL
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv → ReLU → Pool
        x = self.pool(self.relu(self.conv2(x)))  # Conv → ReLU → Pool
        x = x.view(x.size(0), -1)               # Flatten
        x = self.relu(self.fc1(x))               # Fully Connected
        x = self.fc2(x)                          # Output
        return x

# 3. LOSS + OPTIMIZER
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. TRAINING
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/10 | Loss: {running_loss/len(trainloader):.3f}")

# 5. TEST ACCURACY
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Step 6: Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved!")

# Step 7: Save Model
torch.save(model.state_dict(), 'cifar10_model.pth')
print("Model saved!")

# Step 8: Load Model + Predict on new image
from PIL import Image

model2 = SimpleCNN()
model2.load_state_dict(torch.load('cifar10_model.pth'))
model2.eval()

# Predict on test image
image, label = testset[0]
with torch.no_grad():
    output = model2(image.unsqueeze(0))
    predicted = output.argmax(dim=1).item()

print(f"Actual: {classes[label]} | Predicted: {classes[predicted]}")

