import os
import sys
import json
import time
import numpy as np
import torch
import random
import multiprocessing
from typing import Any
from tqdm import tqdm
import engine

MAX_INDICIES = 66

class AnnealedLeakyClippedRelu(torch.nn.Module):
    def __init__(self, leak=0.1):
        super().__init__()
        self.leak = leak

    def forward(self, x):
        # We are modeling an i8 output in the range [0, 127].
        clamped = torch.clamp(x, 0, 127 / 128)
        # Leak a little on either side, but we anneal this to zero.
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
        self.main_net = make_net()
        self.networks = torch.nn.ModuleList([make_net() for _ in range(4 * 8)])
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
        #value = self.main_net(embedding)
        network_outputs = [net(embedding) for net in self.networks]
        ##white_main = self.white_main(embedding)
        ##black_main = self.black_main(embedding)
        ##white_duck = self.white_duck(embedding)
        ##black_duck = self.black_duck(embedding)
        ##data = torch.stack([black_main, white_main, black_duck, white_duck])
        data = torch.stack(network_outputs)
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

def process_one_path(path: str):
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
            pair = e.get_nnue_feature_indices()
            if pair is None:
                continue
            new_feature_indices, net_index = pair
            assert len(new_feature_indices) <= MAX_INDICIES
            state = json.loads(e.get_state())
            examples.append((
                new_feature_indices,
                {"black": 0, "white": 1}[state["turn"]],
                state["isDuckMove"],
                value_for_white,
                net_index,
                move["from"],
                move["to"],
            ))
    print(f"Examples: {len(examples)}")
    indices = -np.ones((len(examples), MAX_INDICIES), dtype=np.int32)
    moves = np.zeros((len(examples), 2), dtype=np.int32)
    meta = np.zeros((len(examples), 5), dtype=np.int32)
    for i, (new_feature_indices, turn, is_duck_move, value_for_white, net_index, move_from, move_to) in enumerate(examples):
        indices[i, :len(new_feature_indices)] = new_feature_indices
        moves[i, :] = move_from, move_to
        meta[i, :] = len(new_feature_indices), value_for_white, turn, is_duck_move, net_index
    np.savez_compressed(result_path, indices=indices, moves=moves, meta=meta)

def process(paths: list[str]):
    pool = multiprocessing.Pool(25)
    pool.map(process_one_path, paths)
    print("Done")

class EWMA:
    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.value = None

    def apply(self, x):
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x

def get_make_batch(paths: list[str], device: str):
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
    all_net_index = concat_meta[:, 4]
    #all_turn = concat_meta[:, 2]
    #all_is_duck_move = concat_meta[:, 3]
    print("Constant model loss:", np.var(all_value_for_white))

    # FIXME: This is a hack to emulate not doing the king-relative embeddings.
    #concat_indices = concat_indices % (13 * 64)

    def make_batch(batch_size, randomize=True):
        if randomize:
            subset = np.random.randint(0, len(concat_indices), size=batch_size)
        else:
            subset = np.arange(batch_size)
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
        which_model = all_net_index[subset]
        #which_model = 2 * all_is_duck_move[subset] + all_turn[subset]
        return (
            torch.tensor(flattened_indices, dtype=torch.int64, device=device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
            torch.tensor(which_model, dtype=torch.int64, device=device),
            torch.tensor(lengths, dtype=torch.int64, device=device),
            torch.tensor(all_value_for_white[subset].reshape((-1, 1)), dtype=torch.float32, device=device),
        )
    return make_batch

def train(paths: list[str], device="cuda"):
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
        lr = max(2e-6, 5e-4 * 0.5 ** (i / 40_000))
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

def quantize(model_path: str, val_path: str, device="cpu"):
    model = Nnue()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    make_batch = get_make_batch([val_path], device)
    val_batch = make_batch(1000)
    # Compute val loss.
    def compute_val_loss():
        with torch.no_grad():
            indices, offsets, which_model, lengths, value_for_white = val_batch
            value_output = model(indices, offsets, which_model, lengths)
            val_loss = torch.nn.MSELoss()(value_output, value_for_white)
        print("Val loss:", val_loss.item())
    compute_val_loss()
    # Each clipped relu wants inputs from -128 to +127 for its active range.
    # If the largest intermediates we care to represent are -2.0 to +2.0,
    # then this means that -2.0 should map to -32768, and +1.99... should map to +32767.
    # Therefore we scale down inputs by 128 before passing them in to the clipped relu.
    # Therefore, 128 * 128 = 16384 represents 1.0 as an input to relu.
    # This means that a quantized weight of 128 represents the weight 1.0 in the original.
    # There is one exception to this, in the original embedding layer, and all biases,
    # where 16384 represents 1.0.
    new_values = {}
    quantized_weights = {}
    output_right_shift = {}
    for k, v in model.named_parameters():
        output_right_shift[k] = 0
        if "main_embed" in k or k == "main_bias":
            quantized = torch.round(v * 2048).to(torch.int16)
            f = quantized.float().detach() / 2048
            output_right_shift[k] = 11
        elif "bias" in k:
            quantized = torch.round(v * 16384).to(torch.int16)
            f = quantized.float().detach() / 16384
            output_right_shift[k] = 14
        else:
            quantized = torch.round(v * 128).to(torch.int8)
            f = quantized.float().detach() / 128
            output_right_shift[k] = 7
        zero_fraction = (quantized == 0).sum() / v.numel()
        new_values[k] = f
        quantized_weights[k] = quantized
        print(f"{k:20} {str(tuple(v.shape)):15} {v.min().item():.3f} {v.max().item():.3f} zero={100 * zero_fraction:.3f}%")
    # Apply the new values.
    model.load_state_dict(new_values)
    compute_val_loss()
    nnue_data = write_nnue_format(
        quantized_weights,
        output_right_shift,
    )
    print(f"NNUE data size: {len(nnue_data) / 1024 / 1024:.2f} MiB")
    with open("nnue.bin", "wb") as f:
        f.write(nnue_data)

def write_nnue_format(
    quantized_weights: dict[str, Any],
    output_right_shift: dict[str, int],
) -> bytes:
    header_alloc = 15 * 1024
    aligned_storage = bytearray(header_alloc)

    def add_bytes(b):
        # Align to the nearest 32-byte boundary.
        padding = (32 - len(aligned_storage)) % 32
        aligned_storage.extend(b'\0' * padding)
        offset = len(aligned_storage)
        aligned_storage.extend(b)
        return offset

    weights = {}
    for k, v in quantized_weights.items():
        shift = output_right_shift[k]
        v = v.detach().cpu().numpy()
        k = k.replace("networks.", "n")
        k = k.replace("0.weight", "0.w")
        k = k.replace("2.weight", "1.w")
        k = k.replace("4.weight", "2.w")
        k = k.replace("0.bias", "0.b")
        k = k.replace("2.bias", "1.b")
        k = k.replace("4.bias", "2.b")
        k = k.replace("main_net.", "n0.")
        offset = add_bytes(v.tobytes())
        assert offset % 32 == 0
        weights[k] = {
            "shape": tuple(v.shape),
            "dtype": {"int8": "i8", "int16": "i16"}[str(v.dtype)],
            "offset": offset,
            "shift": shift,
        }
    message = {
        "version": "v1",
        "weights": weights,
    }

    message_bytes = json.dumps(message).encode()
    assert len(message_bytes) < header_alloc
    aligned_storage[:len(message_bytes)] = message_bytes
    return bytes(aligned_storage)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python train_nnue.py process input1.json input2.json ...")
        print("  python train_nnue.py train input1-nnue-data.npz input2-nnue-data.npz ...")
        print("  python train_nnue.py quantize nnue.pt val-nnue-data.npz")
        exit(1)

    if sys.argv[1] == "process":
        process(sys.argv[2:])
    elif sys.argv[1] == "train":
        train(sys.argv[2:])
    elif sys.argv[1] == "quantize":
        assert len(sys.argv) == 4
        quantize(sys.argv[2], sys.argv[3])
    else:
        print("Unknown command:", sys.argv[1])
        exit(1)
