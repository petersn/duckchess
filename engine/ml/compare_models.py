import sys
import numpy as np
import torch

pytorch_model_path = sys.argv[1]
keras_model_path = sys.argv[2]

# Load up the pytorch model
import model_pytorch

pt_model = model_pytorch.Network().cuda()
pt_model.load_state_dict(torch.load(pytorch_model_path))

# Load up the keras model
import tensorflow as tf

k_model = tf.keras.models.load_model(keras_model_path)

print("Pytorch model:")
print(pt_model)
print("Keras model:")
print(k_model)

inp = np.random.randn(16, 29, 8, 8) * 0.01
pt_inp = torch.tensor(inp).cuda().to(torch.float32)
k_inp = tf.convert_to_tensor(inp)

# Compare model outputs
pt_out = torch.softmax(pt_model(pt_inp)[0], dim=-1).detach().cpu().numpy().flatten()
k_out = k_model(k_inp)[0].numpy().flatten()

print("Pytorch output:")
print(pt_out)
print("Keras output:")
print(k_out)

diff = np.abs(pt_out - k_out)
print("Max diff:", np.max(diff))
print("Avg diff:", np.mean(diff))
