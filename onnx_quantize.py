from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType
)
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTCalibrationReader(CalibrationDataReader):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        self.loader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            data, _ = next(self.iterator)
            return {"input": data.numpy()}
        except StopIteration:
            return None


quantize_static(
    model_input="mnist_fp32.onnx",
    model_output="mnist_int8.onnx",
    calibration_data_reader=MNISTCalibrationReader(),

    # 🔑 THIS IS THE FIX
    activation_type=QuantType.QUInt8,  # activations
    weight_type=QuantType.QInt8        # weights
)

print("INT8 ONNX model created: mnist_int8.onnx")

