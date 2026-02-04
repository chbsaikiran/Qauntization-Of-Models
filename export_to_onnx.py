import torch
from model import ConvNetReLU

# Load FP32 model
model = ConvNetReLU().cpu()
model.load_state_dict(torch.load("mnist_relu_fp32.pth", map_location="cpu"))
model.eval()

# Dummy input (VERY important)
dummy_input = torch.randn(1, 1, 28, 28)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "mnist_fp32.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

print("FP32 ONNX model exported: mnist_fp32.onnx")
print("Done")
