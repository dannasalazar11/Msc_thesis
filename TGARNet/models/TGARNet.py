import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, Input, Reshape, Flatten, Dense, Dropout, Activation, LayerNormalization, BatchNormalization, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import Loss
from keras_nlp.layers import TransformerEncoder
from tensorflow.keras import layers as tf_layers

@tf.keras.utils.register_keras_serializable()
class GaussianKernelLayer(Layer):
    def __init__(self, **kwargs):
        super(GaussianKernelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No es necesario agregar sigma como un peso aquí
        super(GaussianKernelLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs ahora será una lista o tupla: [x, sigma]
        x, sigma = inputs  # Asumimos que sigma viene como entrada junto con los datos
        
        # inputs shape: (N, C, T, F)
        N, C, T, F = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Reshape the input to (N*F, C, T)
        x = tf.transpose(x, perm=(0,3,1,2))  # (N,F,C,T)
        x_reshaped = tf.reshape(x, (N*F, C, T))
        
        # Calculate the pairwise squared Euclidean distance
        squared_differences = tf.expand_dims(x_reshaped, axis=2) - tf.expand_dims(x_reshaped, axis=1)  # (N*F,C,C,T)
        squared_differences = tf.square(squared_differences)  # (N*F,C,C,T)
        pairwise_distances_squared = tf.reduce_sum(squared_differences, axis=-1)  # (N*F,C,C)
        pairwise_distances_squared = tf.reshape(pairwise_distances_squared, (N, F, C, C))  # (N,F,C,C)
        pairwise_distances_squared = tf.transpose(pairwise_distances_squared, perm=(0,2,3,1))  # (N,C,C,F)
        
        # Calculate the Gaussian kernel using the provided sigma
        gaussian_kernel = tf.exp(-pairwise_distances_squared / (2.0 * tf.square(sigma)))
        
        return gaussian_kernel

def renyi_entropy(K, alpha=2):
        """
        input: K tensor, (N,F,C,C)
        output: NxF
        """
        
        C = K.shape[1]
        
        # Normalizamos el kernel antes de calcular la entropía
        
        # Crear una máscara para obtener los elementos diagonales
        diag = tf.expand_dims(tf.linalg.diag_part(K), -1)
        # Calcular el producto de los elementos diagonales
        denominator = tf.math.sqrt(tf.linalg.matmul(diag, diag, transpose_b=True))
        # Normalización
        
        X = (1/C) * tf.math.divide(K, denominator)
        
        if alpha == 2:
            # Realiza el producto matricial entre las dos últimas dimensiones
            X_matmul = tf.linalg.matmul(X, X)
            return -tf.math.log(tf.linalg.trace(X_matmul))
        else:
            # Calcula los autovalores y autovectores de las dos últimas dimensiones
            e, _ = tf.linalg.eigh(X)
            # Calcula la entropía de Renyi
            return (tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, alpha)), axis=-1)) / (1 - alpha))
                               
def joint_renyi_entropy(K, alpha):
        """
        input: K, (N,F,C,C)
        output: Nx1
        """
        
        C = K.shape[-1]
        product = tf.reduce_prod(K,axis=1) # (N,C,C)
        
        trace = tf.linalg.trace(product)
        trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
        trace = tf.tile(trace, [1,C,C])
        
        argument = product/trace
        argument = tf.expand_dims(argument, axis=1) # es necesario porque renyi_entropy recibe 4 dimensiones (1,C,C)
        joint_entropy = renyi_entropy(argument, alpha=alpha)
                               
        return joint_entropy

@tf.keras.utils.register_keras_serializable()
class RenyiMutualInformation(Loss):
    def __init__(self, C, **kwargs):
        self.C = C
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        F = y_pred.shape[1] - 1
        entropy, joint_entropy = tf.split(y_pred, [F, 1], axis=-1)

        entropy = tf.cast(entropy, tf.float64)
        joint_entropy = tf.cast(joint_entropy, tf.float64)
        log_C = tf.math.log(tf.cast(self.C, tf.float64))

        mutual_information = tf.math.abs(
            tf.expand_dims(tf.reduce_sum(entropy, axis=-1), axis=-1) - joint_entropy
        ) / (F * log_C)

        return mutual_information

    def get_config(self):
        config = super().get_config()
        config.update({"C": self.C})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
# Normalizamos 
@tf.keras.utils.register_keras_serializable()
class NormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: N x 2
        y_pred: N x 2 
        """
        batch_size = tf.shape(y_pred)[0]  # batch_size is now an integer tensor
        batch_size_float = tf.cast(batch_size, tf.float32)
        
        cce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        left = tf.tile(tf.expand_dims([1.0, 0.0], axis=0), [batch_size, 1])
        right = tf.tile(tf.expand_dims([0.0, 1.0], axis=0), [batch_size, 1])
        
        cce_left = tf.keras.losses.binary_crossentropy(left, y_pred)
        cce_right = tf.keras.losses.binary_crossentropy(right, y_pred)
        
        cce_norm = tf.divide(cce, (cce_left + cce_right))
        
        return cce_norm

@tf.keras.utils.register_keras_serializable()
class TransposeLayer(Layer):
    def call(self, x):
        return tf.transpose(x, perm = (0,3,1,2))

@tf.keras.utils.register_keras_serializable()        
class RenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha=2, **kwargs):
        super(RenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, K):
        """
        input: K tensor, (N, F, C, C)
        output: NxF
        """
        C = tf.shape(K)[-1]

        # Crear una máscara para obtener los elementos diagonales
        diag = tf.linalg.diag_part(K)
        # Calcular el producto de los elementos diagonales
        denominator = tf.math.sqrt(tf.linalg.matmul(tf.expand_dims(diag, -1), tf.expand_dims(diag, -1), transpose_b=True))
        
        # Normalización
        X = tf.cast((1 / C), tf.float32) * tf.math.divide(K, denominator)

        if self.alpha == 2:
            # Realiza el producto matricial entre las dos últimas dimensiones
            X_matmul = tf.linalg.matmul(X, X)
            return -tf.math.log(tf.linalg.trace(X_matmul))
        else:
            # Calcula los autovalores y autovectores de las dos últimas dimensiones
            e, _ = tf.linalg.eigh(X)
            # Calcula la entropía de Renyi
            return (tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, self.alpha)), axis=-1)) / (1 - self.alpha))


@tf.keras.utils.register_keras_serializable()
class JointRenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super(JointRenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.renyi_entropy_layer = RenyiEntropyLayer(alpha)

    def call(self, K):
        """
        input: K tensor, (N, F, C, C)
        output: Nx1
        """
        C = tf.shape(K)[-1]
        product = tf.reduce_prod(K, axis=1)  # (N, C, C)

        trace = tf.linalg.trace(product)
        trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
        trace = tf.tile(trace, [1, C, C])

        argument = product / trace
        argument = tf.expand_dims(argument, axis=1)  # Necesario porque renyi_entropy recibe 4 dimensiones (1, C, C)
        
        joint_entropy = self.renyi_entropy_layer(argument)  # Llamada a la capa de entropía de Renyi
        return joint_entropy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from keras_nlp.layers import TransformerEncoder
from collections import defaultdict


# --- 1. Clase de Atención Inspeccionable (hereda de la original de Keras) ---
class InspectableMultiHeadAttention(tf_layers.MultiHeadAttention):
    def get_projection_weights(self):
        if not self.built:
            raise ValueError("La capa no ha sido construida.")
        return {
            "query": self._query_dense.kernel.numpy(),
            "key": self._key_dense.kernel.numpy(),
            "value": self._value_dense.kernel.numpy(),
            "output": self._output_dense.kernel.numpy(),
        }
        


# --- 3. Nuestra nueva clase de Encoder Inspeccionable ---
class InspectableTransformerEncoder(TransformerEncoder):
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)

        self._self_attention_layer = InspectableMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            dropout=self.dropout,
            name="self_attention_inspectable"
        )

        self._self_attention_layer_norm = tf_layers.LayerNormalization(epsilon=1e-6)
        self._self_attention_dropout = tf_layers.Dropout(rate=self.dropout)

        self._feedforward_intermediate_dense = tf_layers.Dense(
            self.intermediate_dim,
            activation=self.activation
        )
        self._feedforward_output_dense = tf_layers.Dense(hidden_dim)
        self._feedforward_layer_norm = tf_layers.LayerNormalization(epsilon=1e-6)
        self._feedforward_dropout = tf_layers.Dropout(rate=self.dropout)

        self._last_attention_scores = None

        super(TransformerEncoder, self).build(input_shape)  # no llama al build del padre inmediato
        self.built = True

    def call(self, inputs, padding_mask=None, training=False):
        attention_output, attention_scores = self._self_attention_layer(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True
        )

        self._last_attention_scores = attention_scores

        attention_output = self._self_attention_dropout(attention_output, training=training)
        attention_output = self._self_attention_layer_norm(inputs + attention_output)

        ff_output = self._feedforward_intermediate_dense(attention_output)
        ff_output = self._feedforward_output_dense(ff_output)
        ff_output = self._feedforward_dropout(ff_output, training=training)

        output = self._feedforward_layer_norm(attention_output + ff_output)
        return output

    def get_attention_scores(self):
        if self._last_attention_scores is None:
            raise ValueError("No se han calculado attention scores todavía.")
        return self._last_attention_scores

    def get_attention_weights(self):
        if not self.built:
            raise ValueError("La capa Encoder no ha sido construida.")
        return self._self_attention_layer.get_projection_weights()
        
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Layer


# 🔹 Capa personalizada que combina kernels y expone pesos convexos
class ConvexCombinationLayer(Layer):
    def __init__(self, num_kernels, **kwargs):
        super().__init__(**kwargs)
        self.num_kernels = num_kernels

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(self.num_kernels,),
            initializer=tf.keras.initializers.Constant(1.0 / self.num_kernels),
            trainable=True,
            name="kernel_weights"
        )

    def call(self, inputs):
        weights = tf.nn.softmax(self.alpha)
        weights_reshaped = tf.reshape(weights, (self.num_kernels, 1, 1, 1))
        combined = tf.add_n([w * k for w, k in zip(tf.unstack(weights_reshaped), inputs)])
        # Devuelve ambos: la mezcla combinada y los pesos (broadcast para mantener shape simbólica)
        weights_broadcast = tf.reshape(weights, (1, self.num_kernels))
        return combined, weights_broadcast


def inception_block(x, F, num_kernels, kernel_sigmas):
    kernels = []

    for i in range(0, num_kernels):
        name = f"gaussian_layer_{i+1}"
        sigma = kernel_sigmas[i]
        branch_k = GaussianKernelLayer(name=name)([x, tf.convert_to_tensor(sigma, dtype=tf.float32)])
        kernels.append(branch_k)

    # Combinación convexa con salida dual (combinación + pesos)
    combined_input, kernel_weights_out = ConvexCombinationLayer(num_kernels, name="convex_combination")(kernels)

    # Convolución final
    inception = Conv2D(F, (3, 3), padding='same', activation='relu', name='conv_after_inception')(combined_input)

    # Concatenar los kernels individuales (para la info mutua)
    concatenated_kernels = Concatenate(axis=-1, name='concatenated_kernels')(kernels)

    return concatenated_kernels, inception, kernel_weights_out

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, Input, Reshape, Flatten, Dense, Dropout, Activation, LayerNormalization, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import Loss
from keras_nlp.layers import TransformerEncoder

@tf.keras.utils.register_keras_serializable()
class GaussianKernelLayer(Layer):
    def __init__(self, **kwargs):
        super(GaussianKernelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No es necesario agregar sigma como un peso aquí
        super(GaussianKernelLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs ahora será una lista o tupla: [x, sigma]
        x, sigma = inputs  # Asumimos que sigma viene como entrada junto con los datos
        
        # inputs shape: (N, C, T, F)
        N, C, T, F = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Reshape the input to (N*F, C, T)
        x = tf.transpose(x, perm=(0,3,1,2))  # (N,F,C,T)
        x_reshaped = tf.reshape(x, (N*F, C, T))
        
        # Calculate the pairwise squared Euclidean distance
        squared_differences = tf.expand_dims(x_reshaped, axis=2) - tf.expand_dims(x_reshaped, axis=1)  # (N*F,C,C,T)
        squared_differences = tf.square(squared_differences)  # (N*F,C,C,T)
        pairwise_distances_squared = tf.reduce_sum(squared_differences, axis=-1)  # (N*F,C,C)
        pairwise_distances_squared = tf.reshape(pairwise_distances_squared, (N, F, C, C))  # (N,F,C,C)
        pairwise_distances_squared = tf.transpose(pairwise_distances_squared, perm=(0,2,3,1))  # (N,C,C,F)
        
        # Calculate the Gaussian kernel using the provided sigma
        gaussian_kernel = tf.exp(-pairwise_distances_squared / (2.0 * tf.square(sigma)))
        
        return gaussian_kernel


def renyi_entropy(K, alpha=2):
        """
        input: K tensor, (N,F,C,C)
        output: NxF
        """
        
        C = K.shape[1]
        
        # Normalizamos el kernel antes de calcular la entropía
        
        # Crear una máscara para obtener los elementos diagonales
        diag = tf.expand_dims(tf.linalg.diag_part(K), -1)
        # Calcular el producto de los elementos diagonales
        denominator = tf.math.sqrt(tf.linalg.matmul(diag, diag, transpose_b=True))
        # Normalización
        
        X = (1/C) * tf.math.divide(K, denominator)
        
        if alpha == 2:
            # Realiza el producto matricial entre las dos últimas dimensiones
            X_matmul = tf.linalg.matmul(X, X)
            return -tf.math.log(tf.linalg.trace(X_matmul))
        else:
            # Calcula los autovalores y autovectores de las dos últimas dimensiones
            e, _ = tf.linalg.eigh(X)
            # Calcula la entropía de Renyi
            return (tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, alpha)), axis=-1)) / (1 - alpha))
                               
def joint_renyi_entropy(K, alpha):
        """
        input: K, (N,F,C,C)
        output: Nx1
        """
        
        C = K.shape[-1]
        product = tf.reduce_prod(K,axis=1) # (N,C,C)
        
        trace = tf.linalg.trace(product)
        trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
        trace = tf.tile(trace, [1,C,C])
        
        argument = product/trace
        argument = tf.expand_dims(argument, axis=1) # es necesario porque renyi_entropy recibe 4 dimensiones (1,C,C)
        joint_entropy = renyi_entropy(argument, alpha=alpha)
                               
        return joint_entropy

@tf.keras.utils.register_keras_serializable()
class RenyiMutualInformation(Loss):
    def __init__(self, C, **kwargs):
        self.C = C
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        F = y_pred.shape[1] - 1
        entropy, joint_entropy = tf.split(y_pred, [F, 1], axis=-1)

        entropy = tf.cast(entropy, tf.float64)
        joint_entropy = tf.cast(joint_entropy, tf.float64)
        log_C = tf.math.log(tf.cast(self.C, tf.float64))

        mutual_information = tf.math.abs(
            tf.expand_dims(tf.reduce_sum(entropy, axis=-1), axis=-1) - joint_entropy
        ) / (F * log_C)

        return mutual_information

    def get_config(self):
        config = super().get_config()
        config.update({"C": self.C})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
# Normalizamos 
@tf.keras.utils.register_keras_serializable()
class NormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: N x 2
        y_pred: N x 2 
        """
        batch_size = tf.shape(y_pred)[0]  # batch_size is now an integer tensor
        batch_size_float = tf.cast(batch_size, tf.float32)
        
        cce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        left = tf.tile(tf.expand_dims([1.0, 0.0], axis=0), [batch_size, 1])
        right = tf.tile(tf.expand_dims([0.0, 1.0], axis=0), [batch_size, 1])
        
        cce_left = tf.keras.losses.binary_crossentropy(left, y_pred)
        cce_right = tf.keras.losses.binary_crossentropy(right, y_pred)
        
        cce_norm = tf.divide(cce, (cce_left + cce_right))
        
        return cce_norm

@tf.keras.utils.register_keras_serializable()
class TransposeLayer(Layer):
    def call(self, x):
        return tf.transpose(x, perm = (0,3,1,2))

@tf.keras.utils.register_keras_serializable()        
class RenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha=2, **kwargs):
        super(RenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, K):
        """
        input: K tensor, (N, F, C, C)
        output: NxF
        """
        C = tf.shape(K)[-1]

        # Crear una máscara para obtener los elementos diagonales
        diag = tf.linalg.diag_part(K)
        # Calcular el producto de los elementos diagonales
        denominator = tf.math.sqrt(tf.linalg.matmul(tf.expand_dims(diag, -1), tf.expand_dims(diag, -1), transpose_b=True))
        
        # Normalización
        X = tf.cast((1 / C), tf.float32) * tf.math.divide(K, denominator)

        if self.alpha == 2:
            # Realiza el producto matricial entre las dos últimas dimensiones
            X_matmul = tf.linalg.matmul(X, X)
            return -tf.math.log(tf.linalg.trace(X_matmul))
        else:
            # Calcula los autovalores y autovectores de las dos últimas dimensiones
            e, _ = tf.linalg.eigh(X)
            # Calcula la entropía de Renyi
            return (tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, self.alpha)), axis=-1)) / (1 - self.alpha))


@tf.keras.utils.register_keras_serializable()
class JointRenyiEntropyLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super(JointRenyiEntropyLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.renyi_entropy_layer = RenyiEntropyLayer(alpha)

    def call(self, K):
        """
        input: K tensor, (N, F, C, C)
        output: Nx1
        """
        C = tf.shape(K)[-1]
        product = tf.reduce_prod(K, axis=1)  # (N, C, C)

        trace = tf.linalg.trace(product)
        trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
        trace = tf.tile(trace, [1, C, C])

        argument = product / trace
        argument = tf.expand_dims(argument, axis=1)  # Necesario porque renyi_entropy recibe 4 dimensiones (1, C, C)
        
        joint_entropy = self.renyi_entropy_layer(argument)  # Llamada a la capa de entropía de Renyi
        return joint_entropy
def TGARNet(num_kernels=3, nb_classes=2, Chans=19, Samples=512, 
                                       norm_rate=0.25, alpha=2, num_heads=3, intermediate_dim=128, kernel_sigmas=[1,2,3]):

    input1 = Input(shape=(Chans, Samples))

    # 1 Reorganize data for Transformer (Samples, Chans)
    x = Permute((2, 1))(input1)

    # 2 Normalización antes del Transformer
    x = LayerNormalization()(x)

    # 3 Apply TransformerEncoder
    transformer_encoder = InspectableTransformerEncoder(num_heads=num_heads, intermediate_dim=intermediate_dim, name="transformer_encoder")
    x = transformer_encoder(x)

    # 4 Normalización después del Transformer
    x = LayerNormalization()(x)

    # 5 Restore original shape (Chans, Samples, 1)
    x = Permute((2, 1))(x)
    x = Reshape((Chans, Samples, 1))(x)
    
    # 6 Inception with KernelConv
    concatenated_kernels, inception, kernel_weights = inception_block(x, 5, num_kernels, kernel_sigmas)
    
    # 7 Renyi entropies
    concatenated_kernels = Permute((3, 1, 2))(concatenated_kernels)
    layer_entropy = RenyiEntropyLayer(alpha=alpha)(concatenated_kernels)
    layer_joint_entropy = JointRenyiEntropyLayer(alpha=alpha)(concatenated_kernels)
    entropies_out = Concatenate(axis=-1, name='concatenated_entropies')([
        layer_entropy, layer_joint_entropy
    ])
    
    # 8 Extra convolutional stack
    final_conv = Conv2D(3, kernel_size=3, padding='same', activation='relu', name='Conv2D_2')(inception)
    final_conv = BatchNormalization()(final_conv)
    
    final_conv = Conv2D(3, kernel_size=3, padding='same', activation='relu', name='Conv2D_3')(final_conv)
    final_conv = BatchNormalization()(final_conv)
    
    flat = Flatten()(final_conv)
    drop = Dropout(0.3)(flat)  # Aplica aquí
    dense = Dense(nb_classes, name='output', kernel_constraint=max_norm(norm_rate))(drop)
    softmax = Activation('softmax', name='out_activation')(dense)
    
    model = Model(
        inputs=input1,
        outputs={
            'out_activation': softmax,       # nombre claro
            'entropies_out': entropies_out,   # nombre claro
            'kernel_weights_out': kernel_weights  # nueva salida
        }
    )

    
    return model
