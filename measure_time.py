import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import torch.quantization
from model import ConvNetReLU

def benchmark_fp32(model, input_tensor, runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(input_tensor)

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(input_tensor)
        end = time.perf_counter()

    return (end - start) * 1000 / runs

def benchmark_int8(model, input_tensor, runs=100):
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(input_tensor)

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(input_tensor)
        end = time.perf_counter()

    return (end - start) * 1000 / runs


torch.set_num_threads(1)   # very important for fair comparison
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

data, _ = next(iter(test_loader))
data = data[:1]   # single sample, already normalized

model_fp32 = ConvNetReLU().cpu()
model_fp32.load_state_dict(
    torch.load("mnist_relu_fp32.pth", map_location="cpu")
)
model_fp32.eval()

fp32_time = benchmark_fp32(model_fp32, data)
print(f"FP32 inference time: {fp32_time:.3f} ms")

# ---- INT8 ----
model_int8 = torch.jit.load("mnist_relu_int8.pt", map_location="cpu")

int8_time = benchmark_int8(model_int8, data)
print(f"INT8 inference time: {int8_time:.3f} ms")

