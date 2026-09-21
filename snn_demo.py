import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Setup hardware device and data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 1. Define Network Architecture with Fast Sigmoid Surrogate Gradient
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.95  # Decay factor for membrane potential

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps=25):
        # Flatten image: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        
        # Simulate over time steps (Rate-based temporal encoding)
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            
        return torch.stack(spk2_rec, dim=0)  # Shape: (num_steps, batch, 10)

model = SNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 2. Training Loop
print("Training SNN on PC...")
model.train()
for epoch in range(1):
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        spk_rec = model(data, num_steps=25)
        # Sum spikes over time window for classification target
        spike_counts = spk_rec.sum(dim=0)
        loss = loss_fn(spike_counts, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training complete! Extracting weights for quantization...")

# Extract dynamic range of weights to map float -> int8
w1 = model.fc1.weight.detach().cpu().numpy()
b1 = model.fc1.bias.detach().cpu().numpy()
w2 = model.fc2.weight.detach().cpu().numpy()
b2 = model.fc2.bias.detach().cpu().numpy()