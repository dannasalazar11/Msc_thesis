import tensorflow as tf
from tensorflow.keras import layers, models

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.sequence_length, self.embed_dim),
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embedding


def transformer_block(embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    inputs = layers.Input(shape=(None, embed_dim))
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feedforward network
    ffn_output = layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return models.Model(inputs=inputs, outputs=out2)

def class_of_attention(input_shape=(19, 512),  # 512 time steps, 19 channels
                          num_heads=6,
                          ff_dim=128,
                          num_blocks=6,
                          dense_dim=64,
                          dropout_rate=0.1,
                          num_classes=2):

    inputs = layers.Input(shape=input_shape)

    inputs = layers.Reshape((input_shape[1], input_shape[0]))(inputs)

    x = PositionalEmbedding(sequence_length=input_shape[1], embed_dim=input_shape[0])(inputs)
    
    # Stack multiple transformer blocks
    for _ in range(num_blocks):
        x = transformer_block(embed_dim=input_shape[0], num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)(x)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_dim, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model