import numpy as np
import torch
from tqdm import tqdm

dtype = torch.float32
#dtype = torch.float16

train = np.load("train.npz")
input = torch.tensor(train["input"], dtype=dtype).cuda()
policy = torch.tensor(train["policy"], dtype=dtype).cuda()
value = torch.tensor(train["value"], dtype=dtype).cuda()

policy = policy.reshape((-1, 2 * 8 * 8))
value = value.reshape((-1, 1))

samples = len(input)
assert len(input) == len(policy) == len(value)
print("Loaded", samples, "samples")
print("Value variance:", np.var(train["value"]))

PATCH_SIZE = 128
UPSCALE_COUNT = 1

class ConvBlock(torch.nn.Module):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding="same",
        )
        self.conv2 = torch.nn.Conv2d(
            filters, filters, kernel_size=kernel_size, padding="same"
        )
        self.activation = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        skip0 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x += skip0
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        # x has shape [batch, channel, height, width]
        n, c, h, w = x.shape
        return x.reshape((n, -1))

class Network(torch.nn.Module):
    def __init__(self, blocks, input_channels, feature_count, sm_outputs, sigmoid_outputs):
        super().__init__()
        # Map the three input channels up to our internal features.
        layers = [
            torch.nn.Conv2d(input_channels, feature_count, kernel_size=3, padding="same")
        ]
        # Build the main processing tower.
        for _ in range(blocks):
            layers.append(ConvBlock(feature_count))
        # Globally flatten.
        layers.append(Flatten())
        self.cnn_layers = torch.nn.Sequential(*layers)
        self.sm_linear = torch.nn.Linear(64 * feature_count, sm_outputs)
        self.sigmoid_linear = torch.nn.Linear(64 * feature_count, sigmoid_outputs)

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
    input_channels=input.shape[1],
    feature_count=128,
    sm_outputs=2 * 8 * 8,
    sigmoid_outputs=1,
).cuda()
if dtype == torch.float16:
    model = model.half()

print("Parameter count:", parameter_count(model))

class EWMA:
    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.value = None
        self.all_values = []

    def apply(self, x):
        self.all_values.append(x)
        if self.value is None:
            self.value = x
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * x

def make_batch(batch_size):
    indices = np.random.randint(0, len(input), size=batch_size)
    return (
        input[indices],
        policy[indices],
        value[indices],
    )

# Perform generator pretraining.
cross_en = torch.nn.CrossEntropyLoss()
mse_func = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
sm_loss_ewma = EWMA()
sigmoid_loss_ewma = EWMA()

for i in range(1_000_000):
    optimizer.zero_grad()
    inp, pol, val = make_batch(512)
    sm_output, sigmoid_output = model(inp)
    sm_loss = cross_en(sm_output, pol)
    sigmoid_loss = mse_func(sigmoid_output, val)
    loss = sm_loss + sigmoid_loss
    loss.backward()
    optimizer.step()
    sm_loss_ewma.apply(sm_loss.item())
    sigmoid_loss_ewma.apply(sigmoid_loss.item())

    if i % 100 == 0:
        print("[%7i] loss = %.4f (policy = %.4f  value = %0.4f)" % (
            i, sm_loss_ewma.value + sigmoid_loss_ewma.value, sm_loss_ewma.value, sigmoid_loss_ewma.value,
        ))
    if i % 10_000 == 0:
        torch.save(model.state_dict(), "models/model-001-steps=%i.pt" % i)
