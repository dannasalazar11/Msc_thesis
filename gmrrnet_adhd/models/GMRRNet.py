import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, Input, Reshape, Flatten, Dense, Dropout, Activation, LayerNormalization, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import Loss
from keras_nlp.layers import TransformerEncoder

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

    
def inception_block(x, F, num_kernels):
    kernels = []
    branches = []

    # Genera las ramas y almacénalas en la lista
    for i in range(1, num_kernels + 1):
        name = "gaussian_layer_" + str(i)
        sigma = 5/(2**(i-1))
        branch_k = GaussianKernelLayer(name=name)([x,tf.convert_to_tensor(sigma, dtype=tf.float16)])
        branch = Conv2D(F, (3, 3), padding='same', activation='relu')(branch_k)
        kernels.append(branch_k)
        branches.append(branch)

    # Concatenar todas las ramas usando Concatenate con axis=-1
    concatenated_kernels = Concatenate(axis=-1)(kernels)
    inception = Concatenate(axis=-1)(branches)
    return concatenated_kernels, inception

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
    
class RenyiMutualInformation(Loss):
    def __init__(self, C, **kwargs):
        self.C = C
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: 
        y_pred: N x (F+1) las F entropías marginales y la entropía conjunta
        """
        
        F = y_pred.shape[1]-1
        entropy,  joint_entropy = tf.split(y_pred, [F,1], axis=-1)
        
        #Cast todo
        entropy = tf.cast(entropy, tf.float64)
        joint_entropy = tf.cast(joint_entropy, tf.float64)
        log_C = tf.math.log(tf.cast(self.C, tf.float64))
        
        mutual_information = tf.math.abs((tf.expand_dims(tf.reduce_sum(entropy, axis=-1), axis=-1) - joint_entropy)) / (F * log_C)


        return mutual_information
    
# Normalizamos 

class NormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        y_true: N x 2
        y_pred: N x 2 
        """
        batch_size = tf.shape(y_pred)[0]  # batch_size is now an integer tensor
        batch_size_float = tf.cast(batch_size, tf.float16)
        
        cce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        left = tf.tile(tf.expand_dims([1.0, 0.0], axis=0), [batch_size, 1])
        right = tf.tile(tf.expand_dims([0.0, 1.0], axis=0), [batch_size, 1])
        
        cce_left = tf.keras.losses.binary_crossentropy(left, y_pred)
        cce_right = tf.keras.losses.binary_crossentropy(right, y_pred)
        
        cce_norm = tf.divide(cce, (cce_left + cce_right))
        
        return cce_norm
    
class TransposeLayer(Layer):
    def call(self, x):
        return tf.transpose(x, perm = (0,3,1,2))

class TransposeReshapeLayer(Layer):
    def call(self, x):
        N, C, T, F = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_trans = tf.transpose(x, perm = (0,1,3,2)) # N x C x F x T
        x_resh =  tf.reshape(x_trans, (N, C*F, T))
        return tf.expand_dims(x_resh, -1)
        
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
        X = tf.cast((1 / C), tf.float16) * tf.math.divide(K, denominator)

        if self.alpha == 2:
            # Realiza el producto matricial entre las dos últimas dimensiones
            X_matmul = tf.linalg.matmul(X, X)
            return -tf.math.log(tf.linalg.trace(X_matmul))
        else:
            # Calcula los autovalores y autovectores de las dos últimas dimensiones
            e, _ = tf.linalg.eigh(X)
            # Calcula la entropía de Renyi
            return (tf.math.log(tf.reduce_sum(tf.math.real(tf.math.pow(e, self.alpha)), axis=-1)) / (1 - self.alpha))


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

class TransposeReshapeLayer(tf.keras.layers.Layer):
    def call(self, x):
        N, C, T, F = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_trans = tf.transpose(x, perm=(0,1,3,2))  # (N, C, F, T)
        x_resh = tf.reshape(x_trans, (N, C * F, T))
        return tf.expand_dims(x_resh, -1)

def GMRRNet(num_kernels=3, nb_classes=2, Chans=19, Samples=512, 
                                       norm_rate=0.25, alpha=2, num_heads=3):
    
    input1 = Input(shape=(Chans, Samples))

    # 1 Reorganize data for Transformer (Samples, Chans)
    x = Reshape((Samples, Chans))(input1)

    # 2 Normalización antes del Transformer
    x = LayerNormalization()(x)

    # 3 Apply TransformerEncoder
    transformer_encoder = TransformerEncoder(num_heads=num_heads, intermediate_dim=128)
    x = transformer_encoder(x)

    # 4 Normalización después del Transformer
    x = LayerNormalization()(x)

    # 5 Restore original shape (Chans, Samples, 1)
    x = Reshape((Chans, Samples, 1))(x)
    
    # 6 Inception with KernelConv
    concatenated_branches, inception = inception_block(x, 5, num_kernels)
    
    # 7 Renyi entropies
    concatenated_branches = TransposeLayer()(concatenated_branches)  # => (N, F, C, C)
    layer_entropy = RenyiEntropyLayer(alpha=alpha)(concatenated_branches)
    layer_joint_entropy = JointRenyiEntropyLayer(alpha=alpha)(concatenated_branches)
    entropies_out = Concatenate(axis=-1, name='concatenated_entropies')([
        layer_entropy, layer_joint_entropy
    ])
    
    # 8 Extra convolutional stack
    final_conv = Conv2D(3, kernel_size=3, padding='same', activation='relu', name='Conv2D_2')(inception)
    final_conv = BatchNormalization()(final_conv)
    
    final_conv = Conv2D(3, kernel_size=3, padding='same', activation='relu', name='Conv2D_3')(final_conv)
    final_conv = BatchNormalization()(final_conv)
    
    flat = Flatten()(final_conv)
    dense = Dense(nb_classes, name='output', kernel_constraint=max_norm(norm_rate))(flat)
    dense = Dropout(0.2)(dense)
    softmax = Activation('softmax', name='out_activation')(dense)
    
    model = Model(inputs=input1, outputs=[softmax, entropies_out])
    
    model.compile(
        optimizer='adam',
        loss=[
            'binary_crossentropy',
            None  # For entropy output
        ],
        loss_weights=[0.9, 0.1],
        metrics=[['binary_accuracy'], [None]]
    )
    
    return model
