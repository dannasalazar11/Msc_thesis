import tensorflow as tf
from tensorflow.keras import layers, models

def cnn_lstm_eegnet(
        *,
        n_channels: int = 56,
        n_times: int = 385,
        n_classes: int = 3,
        temporal_filters: int = 50,
        temporal_kernel: int = 50,     # samples → 200 ms @ 256 Hz
        depth_multiplier: int = 2,
        pool_size: int = 40,
        pool_stride: int = 20,
        dropout_rate: float = 0.5,
        lstm_units_1: int = 10,        # paper uses 10 + 10 = 20 total
        lstm_units_2: int = 10,
        activation: str = "elu",
        use_bias_conv: bool = False,
        name: str = "cnn_lstm_eegnet"
    ) -> tf.keras.Model:
    """
    Return an **uncompiled** CNN-LSTM model replicating Wang et al. (2022).

    Change the keyword arguments to vary the architecture.
    """
    act = layers.Activation(activation)

    # ── 1. Input ────────────────────────────────────────────────────
    inp = layers.Input(shape=(n_channels, n_times, 1))

    # ── 2a. Temporal convolution ───────────────────────────────────
    x = layers.Conv2D(filters=temporal_filters,
                      kernel_size=(1, temporal_kernel),
                      padding="same",
                      use_bias=use_bias_conv)(inp)
    x = layers.BatchNormalization()(x)
    x = act(x)

    # ── 2b. Depth-wise spatial convolution ─────────────────────────
    x = layers.DepthwiseConv2D(kernel_size=(n_channels, 1),
                               depth_multiplier=depth_multiplier,
                               use_bias=use_bias_conv)(x)
    x = layers.BatchNormalization()(x)
    x = act(x)

    # ── 2c. Pool + dropout ─────────────────────────────────────────
    x = layers.AveragePooling2D(pool_size=(1, pool_size),
                                strides=(1, pool_stride))(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── 3. LSTM block ──────────────────────────────────────────────
    x = layers.Permute((2, 1, 3))(x)                 # (batch, time, 1, feat)
    x = layers.TimeDistributed(layers.Flatten())(x)  # (batch, time, feat)

    x = layers.LSTM(lstm_units_1, return_sequences=True)(x)
    x = layers.LSTM(lstm_units_2)(x)

    # ── 4. Classifier ──────────────────────────────────────────────
    out = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inp, out, name=name)