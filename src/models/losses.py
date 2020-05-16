import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

class SimilarityLoss(keras.losses.Loss):
    def __init__(self, gamma, beta, alpha,
                 reduction=keras.losses.Reduction.AUTO,
                 name='custom_similarity'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma,
        self.beta  = beta,
        self.alpha = alpha
    
    def __call__(self, pred):
        p0 = pred[:,0,:]
        p1 = pred[:,1,:]
        p2 = pred[:,2,:]
        d1 = K.square(tf.norm(p1-p0, ord='euclidean', axis=1))
        d2 = K.square(tf.norm(p2-p0, ord='euclidean', axis=1))
        return K.maximum(0, self.gamma*d1 - self.beta*d2 + self.alpha)