import tensorflow as tf

def make_model(
    blocks=12,
    feature_count=128,
    final_features=32,
    input_channels=29,
    policy_channels=64,
):
    input_node = tf.keras.Input(shape=(input_channels, 8, 8), name="inp_features")
    x = tf.keras.layers.Conv2D(
        feature_count, 3, padding="same", input_shape=(input_channels, 8, 8), data_format="channels_first",
    )(input_node)
    for _ in range(blocks):
        skip_connection = x
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(feature_count, 3, padding="same", data_format="channels_first")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(feature_count, 3, padding="same", data_format="channels_first")(x)
        x = tf.keras.layers.Add()([skip_connection, x])
    policy = tf.keras.layers.Conv2D(policy_channels, 3, padding="same", data_format="channels_first")(x)
    policy = tf.keras.layers.Flatten()(policy)
    policy = tf.keras.layers.Softmax(name="out_policy")(policy)
    value = tf.keras.layers.Conv2D(1, 3, padding="same", data_format="channels_first")(x)
    value = tf.keras.layers.Flatten()(value)
    value = tf.keras.layers.Dense(1, activation="tanh", name="out_value")(value)
    model = tf.keras.Model(inputs=[input_node], outputs=[policy, value])
    return model
