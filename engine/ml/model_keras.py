import tensorflow as tf

def make_model(
    blocks=12,
    feature_count=128,
    final_features=32,
    input_channels=22,
    policy_outputs=64 * 64,
    value_outputs=1,
):
    input_node = tf.keras.Input(shape=(input_channels, 8, 8), name="inp_features")
    x = tf.keras.layers.Conv2D(
        feature_count, 3, padding="same", input_shape=(input_channels, 8, 8), data_format="channels_first",
    )(input_node)
    for _ in range(blocks):
        skip_connection = x
        x = tf.keras.layers.Conv2D(feature_count, 3, padding="same", activation="relu", data_format="channels_first")(x)
        x = tf.keras.layers.Conv2D(feature_count, 3, padding="same", activation="relu", data_format="channels_first")(x)
        x = tf.keras.layers.Add()([skip_connection, x])
    x = tf.keras.layers.Conv2D(final_features, 3, padding="same", activation="relu", data_format="channels_first")(x)
    x = tf.keras.layers.Flatten()(x)
    policy = tf.keras.layers.Dense(policy_outputs, activation="softmax", name="out_policy")(x)
    value = tf.keras.layers.Dense(value_outputs, activation="tanh", name="out_value")(x)
    model = tf.keras.Model(inputs={"features": input_node}, outputs={"policy": policy, "value": value})
    return model
