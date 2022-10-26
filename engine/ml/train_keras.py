import numpy as np
import tensorflow as tf

train = np.load("train.npz")
train_input = np.moveaxis(train["input"], 1, -1).astype(np.float32)
train_policy = train["policy"]
train_value = train["value"].reshape((-1, 1)).astype(np.float32)
print("SHAPES:")
print("Input:", train_input.shape)
print("Policy:", train_policy.shape)
print("Value:", train_value.shape)

input_node = tf.keras.Input(shape=train_input.shape[1:])
x = tf.keras.layers.Conv2D(
    128, 3, padding="same", input_shape=train_input.shape[1:],
)(input_node)
for _ in range(12):
    skip_connection = x
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Add()([skip_connection, x])
x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.Flatten()(x)
policy = tf.keras.layers.Dense(8 * 8 * 8 * 8, activation="softmax")(x)
value = tf.keras.layers.Dense(1, activation="tanh")(x)

model = tf.keras.Model(inputs=input_node, outputs=[policy, value])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=["sparse_categorical_crossentropy", "mean_squared_error"],
    metrics=["accuracy"],
)
model.summary()

model.fit(
    x=train_input,
    y=[train_policy, train_value],
    batch_size=512,
    epochs=1000,
)
