
import torch

from models.tiny_assembly_ae import (
    TinyAssemblyAE,
    count_parameters
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TinyAssemblyAE().to(device)

model.eval()

print("Device:", device)

print("Parameters:", count_parameters(model))

print(
    "FP32 parameter size:",
    count_parameters(model) * 4 / 1024,
    "KB"
)


# Test different image resolutions

for size in [128, 256, 400]:

    dummy_input = torch.randn(
        1, 3, size, size
    ).to(device)

    with torch.no_grad():

        output = model(dummy_input)

    print("\nInput:", dummy_input.shape)
    print("Output:", output.shape)

    assert output.shape == dummy_input.shape

print("\nMODEL TEST PASSED!")
