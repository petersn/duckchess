import numpy as np
import tensorflow as tf
import torch
import einops

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Limiting on:", gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

input_shape = None, 8, 8, 22

input_node = tf.keras.Input(shape=input_shape[1:], name="inp_features")
x = tf.keras.layers.Conv2D(
    128, 3, padding="same", input_shape=input_shape[1:],
)(input_node)
for _ in range(12):
    skip_connection = x
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Add()([skip_connection, x])
x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)
# We're currently in the shape:  [batch, height, width, channel]
# Before flattening rearrange to [batch, channel, height, width]
x = tf.keras.layers.Permute([3, 1, 2])(x)
x = tf.keras.layers.Flatten()(x)
policy = tf.keras.layers.Dense(8 * 8 * 8 * 8, activation="softmax", name="out_policy")(x)
value = tf.keras.layers.Dense(1, activation="tanh", name="out_value")(x)

model = tf.keras.Model(inputs={"features": input_node}, outputs={"policy": policy, "value": value})
model.compile()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print("Converting:", args.input, "->", args.output)

    state_dict = torch.load(args.input)

    for a, b in zip(state_dict.values(), model.weights):
        a = a.cpu().detach()
        #print(b.name)
        if b.name.startswith("conv2d") and "kernel" in b.name:
            a = einops.rearrange(a, 'feat_out feat_in w h -> w h feat_in feat_out')
        if (b.name.startswith("dense") or b.name.startswith("out_"))and "kernel" in b.name:
            a = einops.rearrange(a, 'a b -> b a')
        print(a.shape, b.shape)
        assert tuple(a.shape) == b.shape, b.name
        b.assign(a)

    model.save(args.output)
