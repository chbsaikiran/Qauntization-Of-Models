import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import torch.quantization
from model import ConvNetReLU

device = "cpu"
print("Using device:", device)

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

#assert torch.backends.quantized.engine == "fbgemm"

model_fp32 = ConvNetReLU().cpu()
model_fp32.load_state_dict(
    torch.load("mnist_relu_fp32.pth", map_location="cpu")
)
model_fp32.eval()

model_fused = copy.deepcopy(model_fp32).cpu().eval()

model_fused = torch.quantization.fuse_modules(
    model_fused,
    [
        ["conv1", "relu1"],
        ["conv2", "relu2"],
        ["fc1", "relu3"]
    ],
    inplace=True
)

model_fused.qconfig = torch.quantization.get_default_qconfig("fbgemm")

torch.backends.quantized.engine = "fbgemm"

model_prepared = torch.quantization.prepare(model_fused)

with torch.no_grad():
    for i, (data, _) in enumerate(train_loader):
        model_prepared(data.cpu())
        if i >= 200:   # enough for MNIST
            break

model_int8_ptq = torch.quantization.convert(model_prepared)
# 🔥 Script it
model_int8_scripted = torch.jit.script(model_int8_ptq)

torch.jit.save(model_int8_scripted, "mnist_relu_int8.pt")

# Quantized inference must be CPU
model_fp32.eval()

data, _ = next(iter(test_loader))
data = data[:1]   # single sample, already normalized

with torch.no_grad():
    y_fp32 = model_fp32(data)
    y_int8 = model_int8_ptq(data)

print("Max abs error:", (y_fp32 - y_int8).abs().max())

def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
    
test(model_int8_ptq, test_loader)


