import numpy as np
import tensorflow as tf
import torch

import model_keras

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print("Converting:", args.input, "->", args.output)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("Limiting on:", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    model = model_keras.make_model()
    model.compile()

    state_dict = torch.load(args.input)

    for a, b in zip(state_dict.values(), model.weights):
        a = a.cpu().detach()
        print(b.name, tuple(a.shape), b.shape)
        assert tuple(a.shape) == b.shape
        b.assign(a)

    model.save(args.output)
