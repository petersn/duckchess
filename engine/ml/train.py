raise NotImplementedError("This file is deprecated")

import numpy as np
import torch
from tqdm import tqdm

dtype = torch.float32
#dtype = torch.float16

train = np.load("train.npz")
input = torch.tensor(train["input"])#, dtype=dtype)
policy = torch.tensor(train["policy"])#, dtype=torch.int64)
value = torch.tensor(train["value"])#, dtype=dtype)

value = value.reshape((-1, 1))

samples = len(input)
assert len(input) == len(policy) == len(value)
print("Loaded", samples, "samples")
print("Value variance:", np.var(train["value"]))

model = Network(
    blocks=12,
    input_channels=input.shape[1],
    feature_count=128,
    final_features=4,
    sm_outputs=64 * 64,
    tanh_outputs=1,
).cuda()
if dtype == torch.float16:
    model = model.half()

def parameter_count(model):
    return sum(np.prod(p.size()) for p in model.parameters())

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
        input[indices].cuda().to(dtype),
        policy[indices].cuda().to(torch.int64),
        value[indices].cuda().to(dtype),
    )

# Perform generator pretraining.
cross_en = torch.nn.CrossEntropyLoss()
mse_func = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
sm_loss_ewma = EWMA()
tanh_loss_ewma = EWMA()

# Load up the model
#model.load_state_dict(torch.load("models/model-001-steps=140000.pt"))
#model.load_state_dict(torch.load("models/model-001-steps=350000.pt"))
#model.load_state_dict(torch.load("models/model-001-steps=850000.pt"))

for i in range(10_000_000):
#for i in range(140_001, 1_000_000):
#for i in range(350_001, 1_000_000):
#for i in range(440_001, 10_000_000):
    optimizer.zero_grad()
    inp, pol, val = make_batch(1024)
    sm_output, tanh_output = model(inp)
    sm_loss = cross_en(sm_output, pol)
    tanh_loss = mse_func(tanh_output, val)
    loss = sm_loss + tanh_loss
    loss.backward()
    optimizer.step()
    sm_loss_ewma.apply(sm_loss.item())
    tanh_loss_ewma.apply(tanh_loss.item())

    if i % 100 == 0:
        print("[%7i] loss = %.4f (policy = %.4f  value = %0.4f)" % (
            i, sm_loss_ewma.value + tanh_loss_ewma.value, sm_loss_ewma.value, tanh_loss_ewma.value,
        ))
    if i % 10_000 == 0:
        torch.save(model.state_dict(), "models/model-002-steps=%i.pt" % i)
