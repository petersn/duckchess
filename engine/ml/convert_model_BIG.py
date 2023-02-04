import os
import subprocess
import numpy as np
import tensorflow as tf
import torch
import einops

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--big", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--tfjs", action="store_true")
    args = parser.parse_args()

    print("Converting:", args.input, "->", args.output)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        print("Limiting on:", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.big:
        import model_keras_BIG as model_keras
    elif args.medium:
        import model_keras_MEDIUM as model_keras
    else:
        import model_keras

    model = model_keras.make_model()
    model.compile()

    weights_list = list(model.weights)
    # FIXME: This little hacky fixup is absurdly brittle!
    # Move the two conv2d_42 layers forward by one.
    if args.big:
        i = [w.name.startswith("conv2d_22/") for w in weights_list].index(True)
        print("First conv2d_22 index:", i)
        weights_list[i : i + 2], weights_list[i + 2 : i + 4] = weights_list[i + 2 : i + 4], weights_list[i : i + 2]
    if args.medium:
        i = [w.name.startswith("conv2d_22/") for w in weights_list].index(True)
        print("First conv2d_22 index:", i)
        weights_list[i : i + 2], weights_list[i + 2 : i + 4] = weights_list[i + 2 : i + 4], weights_list[i : i + 2]


    state_dict = torch.load(args.input)
    for (torch_name, torch_weight), keras_weight in zip(state_dict.items(), weights_list):
        print(torch_name, keras_weight.name, tuple(torch_weight.shape), keras_weight.shape)
    # print("LENGTHS:", len(state_dict.values()), len(weights_list))
    # # Print the excess elements in whoever is longer.
    # if len(state_dict.values()) > len(weights_list):
    #     for torch_weight in list(state_dict.values())[len(weights_list) :]:
    #         print(torch_weight.name, tuple(torch_weight.shape))
    # elif len(weights_list) > len(state_dict.values()):
    #     for keras_weight in weights_list[len(state_dict.values()) :]:
    #         print(keras_weight.name, keras_weight.shape)
    # assert len(state_dict.values()) == len(weights_list)
    print("====================")

    for torch_weight, keras_weight in zip(state_dict.values(), weights_list):
        # Torch stores conv kernels in shape: [outs, ins, width, height]
        # Keras stores conv kernels in shape: [width, height, ins, outs]
        torch_weight = torch_weight.cpu().detach()
        if keras_weight.name.startswith("conv2d") and "kernel" in keras_weight.name:
            torch_weight = einops.rearrange(torch_weight, "outs ins w h -> w h ins outs")
        if (keras_weight.name.startswith("dense") or keras_weight.name.startswith("out_"))and "kernel" in keras_weight.name:
            torch_weight = einops.rearrange(torch_weight, "a b -> b a")
        print(torch_weight.name, keras_weight.name, tuple(torch_weight.shape), keras_weight.shape)
        assert tuple(torch_weight.shape) == keras_weight.shape
        keras_weight.assign(torch_weight)

    if args.tfjs:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, args.output)
        print("Saved TFJS model:", args.output)
    else:
        model.save(args.output + "-keras")
        print("Saved Keras model:", args.output + "-keras")
        # Convert the Keras model to ONNX.
        subprocess.check_call([
            "python", "-m", "tf2onnx.convert",
            "--saved-model", args.output + "-keras",
            "--output", args.output + ".onnx",
            "--opset", "11",
            "--verbose",
            "--target", "tensorrt",
        ])
        print("Saved ONNX model:", args.output + ".onnx")
        # Convert the ONNX to a TensorRT engine.
        # FIXME: This is turbo-hacky, but I need to generate one plan per GPU type,
        # to avoid issues with compute capabilities on loading.
        for compute_capability, gpu_index in [("compute8.6", 0)]: #[("compute8.9", 0), ("compute8.6", 1)]:
            output_path = args.output + "-" + compute_capability + ".trt"
            subprocess.check_call(
                [
                    "trtexec",
                    "--fp16",
                    "--onnx=" + args.output + ".onnx",
                    "--shapes=inp_features:128x37x8x8",
                    "--saveEngine=" + output_path,
                    "--timingCacheFile=trtexec-timing-cache",
                ],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index)),
            )
            print("Saved TensorRT engine:", output_path)
