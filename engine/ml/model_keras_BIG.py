import tensorflow as tf

def make_model(
    blocks=10,
    feature_count=128,
    final_features=64,
    input_channels=37,
    policy_channels=64,
    value_channels=8,
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
    value = tf.keras.layers.Conv2D(value_channels, 3, padding="same", data_format="channels_first")(x)
    value = tf.keras.layers.Activation("relu")(value)
    value = tf.keras.layers.Flatten()(value)
    value = tf.keras.layers.Dense(final_features, activation="relu")(value)
    wdl = tf.keras.layers.Dense(3, name="out_wdl")(value)
    mcts_value_prediction = tf.keras.layers.Dense(1, activation="tanh", name="out_mcts_value_prediction")(value)
    model = tf.keras.Model(inputs=[input_node], outputs=[policy, wdl, mcts_value_prediction])
    return model
