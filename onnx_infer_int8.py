import onnxruntime as ort
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load INT8 model
session = ort.InferenceSession(
    "mnist_int8.onnx",
    providers=["CPUExecutionProvider"]
)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
data, _ = next(iter(loader))
data_np = data.numpy()

# Warmup
for _ in range(10):
    session.run(None, {"input": data_np})

# Benchmark
runs = 100
start = time.perf_counter()
for _ in range(runs):
    session.run(None, {"input": data_np})
end = time.perf_counter()

print(f"ONNX INT8 inference time: {(end - start)*1000/runs:.3f} ms")

