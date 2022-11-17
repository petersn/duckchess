import os
import sys
import json
import time
import numpy as np
import torch
import random
import multiprocessing
from tqdm import tqdm
import engine

MAX_INDICIES = 66

class AnnealedLeakyClippedRelu(torch.nn.Module):
    def __init__(self, leak=0.1):
        super().__init__()
        self.leak = leak

    def forward(self, x):
        # We are modeling and i8 output in the range [-128, 127].
        clamped = torch.clamp(x, -1, 127 / 128)
        # Leak a little on either side.
        return clamped + self.leak * (x - clamped)

class Nnue(torch.nn.Module):
    # For each of 128 possible black or white king positions we have:
    #   * six white pieces * 64 squares
    #   * six black pieces * 64 squares
    #   * the duck * 64 squares

    #FEATURE_COUNT = 2 * 64 * ((5 + 5 + 1) * 64) + 64 + 64
    FEATURE_COUNT = 2 * 64 * ((6 + 6 + 1) * 64)
    ACCUM_SIZE = 256

    def __init__(self):
        super().__init__()
        # We output ACCUM_SIZE intermediate values, and one PSQT output that bypasses the net.
        self.main_embed = torch.nn.EmbeddingBag(
            num_embeddings=self.FEATURE_COUNT,
            embedding_dim=self.ACCUM_SIZE,
            mode="sum",
        )
        # Scale down the embedding weights by a lot so that rare values do very little.
        self.main_embed.weight.data /= 100.0

        self.main_bias = torch.nn.Parameter(torch.zeros(self.ACCUM_SIZE))
        self.clipped_relu = AnnealedLeakyClippedRelu()
        self.tanh = torch.nn.Tanh()
        make_net = lambda: torch.nn.Sequential(
            torch.nn.Linear(self.ACCUM_SIZE, 16),
            self.clipped_relu,
            torch.nn.Linear(16, 32),
            self.clipped_relu,
            torch.nn.Linear(32, 1),
        )
        self.networks = torch.nn.ModuleList([make_net() for _ in range(4 * 4)])
        #self.white_main = make_net()
        #self.black_main = make_net()
        #self.white_duck = make_net()
        #self.black_duck = make_net()

    def adjust_leak(self, new_leak):
        self.clipped_relu.leak = new_leak

    def forward(self, indices, offsets, which_model, lengths):
        accum = self.main_embed(indices, offsets) + self.main_bias
        psqt = accum[:, :1]
        embedding = self.clipped_relu(accum)
        network_outputs = [net(embedding) for net in self.networks]
        #white_main = self.white_main(embedding)
        #black_main = self.black_main(embedding)
        #white_duck = self.white_duck(embedding)
        #black_duck = self.black_duck(embedding)
        #data = torch.stack([black_main, white_main, black_duck, white_duck])
        data = torch.stack(network_outputs)
        game_phase = torch.div(lengths, 17, rounding_mode="floor")
        which_model = which_model + game_phase * 4
        value = data[which_model, torch.arange(len(which_model))]
        return self.tanh(value + psqt)

class PieceSquareTable(torch.nn.Module):
    PIECE_SQUARE_POSITIONS = 2 * 64 * ((6 + 6 + 1) * 64) #13 * 64

    def __init__(self):
        super().__init__()
        self.table = torch.nn.EmbeddingBag(
            num_embeddings=self.PIECE_SQUARE_POSITIONS,
            embedding_dim=1,
            mode="sum",
        )
        self.tanh = torch.nn.Tanh()

    def adjust_leak(self, new_leak):
        pass

    def forward(self, indices, offsets, which_model, lengths):
        value = self.table(indices, offsets)
        return self.tanh(value)

def process_one_path(path):
    assert path.endswith(".json")
    result_path = path[:-5] + "-nnue-data.npz"
    if os.path.exists(result_path):
        print("Skipping", path)
        return

    print(f"Processing: {path}")

    examples = []
    lines = open(path).readlines()
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
            assert len(new_feature_indices) <= MAX_INDICIES
            state = json.loads(e.get_state())
            examples.append((
                new_feature_indices,
                {"black": 0, "white": 1}[state["turn"]],
                state["isDuckMove"],
                value_for_white,
            ))
    print(f"Examples: {len(examples)}")
    indices = -np.ones((len(examples), MAX_INDICIES), dtype=np.int32)
    meta = np.zeros((len(examples), 4), dtype=np.int32)
    for i, (new_feature_indices, turn, is_duck_move, value_for_white) in enumerate(examples):
        indices[i, :len(new_feature_indices)] = new_feature_indices
        meta[i, :] = len(new_feature_indices), value_for_white, turn, is_duck_move
    np.savez_compressed(result_path, indices=indices, meta=meta)

def process(paths):
    pool = multiprocessing.Pool(14)
    pool.map(process_one_path, paths)
    print("Done")

class EWMA:
    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.value = None

    def apply(self, x):
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x

def get_make_batch(paths, device):
    total_examples = 0
    all_indices = []
    all_meta = []
    for path in tqdm(paths):
        assert path.endswith(".npz")
        data = np.load(path)
        all_indices.append(data["indices"])
        all_meta.append(data["meta"])
    concat_indices = np.concatenate(all_indices)
    concat_meta = np.concatenate(all_meta)
    assert concat_indices.shape[0] == concat_meta.shape[0]
    print(f"Total examples: {len(concat_indices)}")
    all_indices_length = concat_meta[:, 0]
    all_value_for_white = concat_meta[:, 1]
    all_turn = concat_meta[:, 2]
    all_is_duck_move = concat_meta[:, 3]
    print("Constant model loss:", np.var(all_value_for_white))

    # FIXME: This is a hack to emulate not doing the king-relative embeddings.
    #concat_indices = concat_indices % (13 * 64)

    def make_batch(batch_size):
        subset = np.random.randint(0, len(concat_indices), size=batch_size)
        lengths = all_indices_length[subset]
        total = np.sum(lengths)
        flattened_indices = np.zeros(total, dtype=np.int64)
        offsets = np.zeros(batch_size, dtype=np.int64)
        next_offset = 0
        for i, length in enumerate(lengths):
            offsets[i] = next_offset
            flattened_indices[next_offset:next_offset + length] = concat_indices[subset[i], :length]
            next_offset += length
        assert next_offset == total
        which_model = 2 * all_is_duck_move[subset] + all_turn[subset]
        return (
            torch.tensor(flattened_indices, dtype=torch.int64, device=device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
            torch.tensor(which_model, dtype=torch.int64, device=device),
            torch.tensor(lengths, dtype=torch.int64, device=device),
            torch.tensor(all_value_for_white[subset].reshape((-1, 1)), dtype=torch.float32, device=device),
        )
    return make_batch

def train(paths, device="cuda"):
    print(f"Training on {len(paths)} files on {device}")
    # Begin a training loop.
    model = Nnue().to(device)
    #model = PieceSquareTable().to(device)
    print("Parameters:", sum(np.product(t.shape) for t in model.parameters()))
    mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    loss_ewma = EWMA()

    make_batch = get_make_batch(paths, device)

    start_time = time.time()
    for i in range(1_000_000):
        leak = 0.1 * 0.5 ** (i / 10_000)
        lr = max(5e-6, 5e-4 * 0.5 ** (i / 20_000))
        #lr = 1e-3
        model.adjust_leak(leak)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()
        indices, offsets, which_model, lengths, value_for_white = make_batch(256)
        value_output = model(indices, offsets, which_model, lengths)
        loss = mse_func(value_output, value_for_white)
        loss.backward()
        optimizer.step()
        loss_ewma.apply(loss.item())

        if i % 1000 == 0:
            print("(%7.1f) [%7i] loss = %.4f  leak = %.4f  lr = %.6f" % (
                time.time() - start_time,
                i,
                loss_ewma.value,
                leak,
                lr,
            ))
            torch.save(model.state_dict(), "nnue.pt")

def quantize(model_path):
    model = Nnue()
    model.load_state_dict(torch.load(model_path))
    print(model)
    


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python train_nnue.py process input1.json input2.json ...")
        print("  python train_nnue.py train input1-nnue-data.npz input2-nnue-data.npz ...")
        print("  python train_nnue.py quantize nnue.pt")
        exit(1)

    if sys.argv[1] == "process":
        process(sys.argv[2:])
    elif sys.argv[1] == "train":
        train(sys.argv[2:])
    elif sys.argv[1] == "quantize":
        quantize(sys.argv[2])
    else:
        print("Unknown command:", sys.argv[1])
        exit(1)
