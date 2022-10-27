import numpy as np
import tensorflow as tf
import torch
import einops

import model_keras

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print("Converting:", args.input, "->", args.output)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        print("Limiting on:", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    model = model_keras.make_model()
    model.compile()

    state_dict = torch.load(args.input)

    for torch_weight, keras_weight in zip(state_dict.values(), model.weights):
        # Torch stores conv kernels in shape: [outs, ins, width, height]
        # Keras stores conv kernels in shape: [width, height, ins, outs]
        torch_weight = torch_weight.cpu().detach()
        if keras_weight.name.startswith("conv2d") and "kernel" in keras_weight.name:
            torch_weight = einops.rearrange(torch_weight, "outs ins w h -> w h ins outs")
        if (keras_weight.name.startswith("dense") or keras_weight.name.startswith("out_"))and "kernel" in keras_weight.name:
            torch_weight = einops.rearrange(torch_weight, "a b -> b a")
        print(keras_weight.name, tuple(torch_weight.shape), keras_weight.shape)
        assert tuple(torch_weight.shape) == keras_weight.shape
        keras_weight.assign(torch_weight)

    model.save(args.output)
