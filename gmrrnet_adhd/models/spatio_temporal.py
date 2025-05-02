import os
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, models

import numpy as np
from scipy.signal import welch, butter, filtfilt

# ---------------------------------------------------------------------------
# 1. Positional encoding (fixed sinusoidal, Vaswani et al. 2017)
# ---------------------------------------------------------------------------

class PositionalEncoding(layers.Layer):
    """Adds 1-D sinusoidal positional embeddings to a sequence tensor."""

    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        # positions × channels → (max_len, d_model)
        pos = tf.cast(tf.range(max_len)[:, tf.newaxis], tf.float32)  # (L,1)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)    # (1,D)

        exponent = (2.0 * (i // 2)) / tf.cast(d_model, tf.float32)   # float32
        angle_rates = tf.constant(1.0, tf.float32) / tf.pow(10000.0, exponent)
        angle_rads = pos * angle_rates                               # (L,D)

        # Even indices → sin, odd indices → cos
        sin = tf.sin(angle_rads[:, 0::2])
        cos = tf.cos(angle_rads[:, 1::2])
        pe = tf.reshape(tf.concat([sin, cos], axis=-1), (max_len, d_model))
        self.pos_encoding = pe[tf.newaxis]                           # (1,L,D)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return tf.cast(x, tf.float32) + self.pos_encoding[:, :seq_len, :]


# ---------------------------------------------------------------------------
# 2. Transformer encoder block (MHA + FFN)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128,
                 rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = models.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.attn(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ---------------------------------------------------------------------------
# 3. Stream builder (spectral / temporal / spatial)
# ---------------------------------------------------------------------------

def build_stream(name: str, seq_len: int, feat_dim: int, d_model: int = 64,
                 num_layers: int = 2) -> Tuple[Input, layers.Layer]:
    """Create one transformer stream."""
    inp = Input(shape=(seq_len, feat_dim), name=f"{name}_input")
    x = layers.Dense(d_model)(inp)              # linear projection
    x = PositionalEncoding(d_model)(x)          # add PE
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model=d_model)(x)
    x = layers.GlobalAveragePooling1D()(x)      # sequence → vector
    x = layers.Dense(128, activation="relu")(x)  # per-stream decoder
    return inp, x


# ---------------------------------------------------------------------------
# 4. Full multi-stream model
# ---------------------------------------------------------------------------

def build_eeg_attention_model(
    freq_shape: Tuple[int, int],
    temp_shape: Tuple[int, int],
    spat_shape: Tuple[int, int],
    d_model: int = 64,
    num_layers: int = 2,
) -> Model:
    """Assemble the three-stream transformer network."""
    freq_in, freq_vec = build_stream("freq", *freq_shape, d_model=d_model, num_layers=num_layers)
    temp_in, temp_vec = build_stream("temp", *temp_shape, d_model=d_model, num_layers=num_layers)
    spat_in, spat_vec = build_stream("spat", *spat_shape, d_model=d_model, num_layers=num_layers)

    concat = layers.concatenate([freq_vec, temp_vec, spat_vec], name="concat_streams")
    x = layers.Dense(128, activation="relu")(concat)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(2, activation="softmax", name="attention_state")(x)

    return Model(inputs=[freq_in, temp_in, spat_in], outputs=output, name="EEG_Attention_Transformer")


def band_power_psd(x, fs, bands):
    """Compute average power in each frequency band for one trial."""
    freqs, psd = welch(x, fs=fs, nperseg=fs*2)          # (freq,)
    pow_in_band = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        pow_in_band.append(psd[..., mask].mean())
    return np.array(pow_in_band)                         # (n_bands,)

import numpy as np
from scipy.signal import welch

def prepare_streams_4s(X, fs=128, n_win=10):
    """
    X : ndarray (N, C, 512)   4‑second segments @128 Hz
    Returns freq, temp, spat ready for model.fit
    """
    N, C, T = X.shape
    assert T == 512, "Expecting exactly 512‑sample segments (4 s @128 Hz)"

    # ── 1. Spectral stream (20 log bands) ────────────────────────
    log_bands = np.logspace(np.log10(1), np.log10(fs / 2), 21)
    bands = list(zip(log_bands[:-1], log_bands[1:]))

    def band_power(signal):
        f, Pxx = welch(signal, fs=fs, nperseg=512)     # 512‑pt FFT → all bands covered
        return np.array([
            Pxx[(f >= lo) & (f < hi)].mean()
            if np.any((f >= lo) & (f < hi)) else 0.0   # avoid empty‑slice warning
            for lo, hi in bands
        ])

    freq_feats = np.stack([[band_power(ch) for ch in trial] for trial in X])
    freq = freq_feats.mean(axis=1)[..., None]          # (N, 20, 1)

    # ── 2. Temporal stream (10 × 51 samples = 510 samples) ──────
    T_win = T // n_win           # 512 // 10 = 51
    usable = n_win * T_win       # 510
    x_trim = X[:, :, :usable]    # drop the last 2 samples
    windows = x_trim.reshape(N, C, n_win, T_win)   # (N, C, 10, 51)

    # simple mean amplitude; swap in richer stats if you like
    temp = windows.mean(axis=(1, 3))[..., None]        # (N, 10, 1)

    # ── 3. Spatial stream (per‑channel RMS) ─────────────────────
    spat = np.sqrt((X ** 2).mean(axis=2))[..., None]   # (N, C, 1)

    return (freq.astype("float32"),
            temp.astype("float32"),
            spat.astype("float32"))

