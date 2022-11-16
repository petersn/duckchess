import sys
import json
import numpy as np
import torch
import random
from tqdm import tqdm
import engine

class AnnealedLeakyClippedRelu(torch.nn.Module):
    def __init__(self, leak=0.05):
        super().__init__()
        self.leak = leak

    def forward(self, x):
        # We are modeling and i8 output in the range [-128, 127].
        clamped = torch.clamp(x, -1, 127 / 128)
        # Leak a little on either side.
        return clamped + self.leak * (x - clamped)

class Nnue(torch.nn.Module):
    # For each of 128 possible black or white king positions we have:
    #   * five white non-king pieces * 64 squares
    #   * five black non-king pieces * 64 squares
    #   * the duck * 64 squares
    # Finally, we have 128 additional biases.

    FEATURE_COUNT = 2 * 64 * ((5 + 5 + 1) * 64) + 64 + 64
    ACCUM_SIZE = 256

    def __init__(self):
        super().__init__()
        # We output ACCUM_SIZE intermediate values, and one PSQT output that bypasses the net.
        self.main_embed = torch.nn.EmbeddingBag(self.FEATURE_COUNT, self.ACCUM_SIZE + 1)
        self.clipped_relu = AnnealedLeakyClippedRelu()
        self.tanh = torch.nn.Tanh()
        make_net = lambda: torch.nn.Sequential(
            torch.nn.Linear(self.ACCUM_SIZE, 16),
            self.clipped_relu,
            torch.nn.Linear(16, 32),
            self.clipped_relu,
            torch.nn.Linear(32, 1),
        )
        self.white_main = make_net()
        self.black_main = make_net()
        self.white_duck = make_net()
        self.black_duck = make_net()

    def forward(self, inputs, which_model):
        embedding = self.main_embed(inputs)
        embedding = self.clipped_relu(embedding)
        white_main = self.white_main(embedding)
        black_main = self.black_main(embedding)
        white_duck = self.white_duck(embedding)
        black_duck = self.black_duck(embedding)
        data = torch.stack([white_main, black_main, white_duck, black_duck])
        value = data[which_model, torch.arange(len(which_model))]
        return self.tanh(value)

model = Nnue()

print("Parameters:", sum(np.product(t.shape) for t in model.parameters()))

lines = [
    line
    for path in tqdm(sys.argv[1:])
    for line in open(path)
    if line.strip()
]
print("Games:", len(lines))

feature_indices = []
bag_offsets = [0]
train_targets = []

def add_example(new_feature_indices, value_for_white):
    feature_indices.extend(new_feature_indices)
    bag_offsets.append(len(feature_indices))
    train_targets.append(value_for_white)

for line in tqdm(lines):
    game = json.loads(line)
    e = engine.Engine(0)
    value_for_white = {"1-0": +1, "0-1": -1, None: 0}[game["outcome"]]
    for move in game["moves"]:
        move_str = json.dumps(move)
        r = e.apply_move(move_str)
        assert r is None, f"Move {move_str} failed: {r}"
        new_feature_indices = e.get_nnue_feature_indices()
        if new_feature_indices is None:
            continue
        add_example(new_feature_indices, value_for_white)
bag_offsets.pop()

print("Total features:", len(feature_indices))
print("Total examples:", len(train_targets))
