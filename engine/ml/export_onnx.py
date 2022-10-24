import numpy as np
import torch

PAD = 1

class ConvBlock(torch.nn.Module):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding=PAD,
        )
        self.conv2 = torch.nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding=PAD
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        skip0 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x + skip0
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        # x has shape [batch, channel, height, width]
        n, c, h, w = x.shape
        return x.reshape((n, -1))

class Network(torch.nn.Module):
    def __init__(
        self,
        blocks,
        input_channels,
        feature_count,
        final_features,
        sm_outputs,
        sigmoid_outputs,
    ):
        super().__init__()
        # Map the three input channels up to our internal features.
        layers = [
            torch.nn.Conv2d(input_channels, feature_count, kernel_size=3, padding=PAD),
        ]
        # Build the main processing tower.
        for _ in range(blocks):
            layers.append(ConvBlock(feature_count))
        layers.extend([
            torch.nn.Conv2d(feature_count, final_features, kernel_size=3, padding=PAD),
            torch.nn.ReLU(),
        ])
        # Globally flatten.
        layers.append(Flatten())
        self.cnn_layers = torch.nn.Sequential(*layers)
        self.sm_linear = torch.nn.Linear(64 * final_features, sm_outputs)
        self.sigmoid_linear = torch.nn.Linear(64 * final_features, sigmoid_outputs)

    def forward(self, x):
        deep_embedding = self.cnn_layers(x)
        sm_output = self.sm_linear(deep_embedding)
        sigmoid_output = self.sigmoid_linear(deep_embedding)
        sigmoid_output = torch.sigmoid(sigmoid_output)
        return sm_output, sigmoid_output

def parameter_count(model):
    return sum(np.prod(p.size()) for p in model.parameters())

model = Network(
    blocks=12,
    input_channels=22,
    feature_count=128,
    final_features=4,
    sm_outputs=64 * 64,
    sigmoid_outputs=1,
)
model.eval()

example = torch.tensor(np.random.randn(1, 22, 8, 8), dtype=torch.float32)

import torch.onnx

torch.onnx.export(model,               # model being run
                  example,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['policy', 'value'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'policy' : {0 : 'batch_size'}, 'value': {0: 'batch_size'}})


#traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save("model-jit.pt")

