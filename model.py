import torch
import torch.nn as nn
import torch.quantization as tq

class ConvNetReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 👇 REQUIRED for PTQ / QAT
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.quant(x)

        x = self.relu1(self.conv1(x))
        x = self.pool(self.relu2(self.conv2(x)))

        x = x.reshape(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)
        return x
