import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data,shuffle = True,batch_size=64)
test_loader = DataLoader(test_data, shuffle = False, batch_size=64)

class DigiNet(nn.Module):
    def __init__(self):
        super(DigiNet,self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

model = DigiNet()
print(model)

optimiser = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 3

for epoch in range(epochs):
    model.train()

    for images,labels in train_loader:
        optimiser.zero_grad()
        preds=model(images)
        loss = loss_fn(preds,labels)
        loss.backward()
        optimiser.step()
    print(f"Epoch {epoch+1} done, last batch loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images,labels in test_loader:
        preds = model(images)
        predicted_digit = preds.argmax(dim=1)
        correct += (predicted_digit == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test accuracy: {accuracy:.2f}%")