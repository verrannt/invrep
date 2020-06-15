import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

class OnlineTripletLoss(keras.losses.Loss):
    def __init__(self, margin=1.0, squared=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='online_triplet_loss'):
        super().__init__(reduction=reduction, name=name)
        self.margin = margin
        self.squared = squared
    
    def __call__(self, y_true, y_pred):
        """
        Params:
        =======
        y_true : array of labels, can be one-hot encoded
            (dim: batchsize, #classes)
        y_pred : the embeddings of the network
            (dim: batchsize, embedding shape)
        """
        loss = self._get_distance_tensor(y_pred)

        Rt = self._get_Rt(y_true)

        loss = tf.maximum(tf.multiply(Rt, loss+self.margin), 0.0)

        n_valid_triplets = tf.reduce_sum(tf.abs(Rt))
        
        # Count number of positive losses
        #num_positive_triplets = tf.reduce_sum(
        #    tf.cast(tf.greater(loss, 1e-16), tf.float32))

        # Get final mean triplet loss over all valid triplets
        loss = tf.reduce_sum(loss) / (n_valid_triplets) #(num_positive_triplets + 1e-16)

        return loss
    
    def _get_distance_tensor(self, y_pred):
            y_pred = self._get_pairwise_distances(y_pred, squared=self.squared)
            pred_i_and_j = tf.expand_dims(y_pred, 2)
            pred_i_and_k = tf.expand_dims(y_pred, 1)
            loss = tf.math.subtract(pred_i_and_j, pred_i_and_k)
            return loss

    def _get_Rt(self, labels):
        """Return a 3-D mask where 
            mask[a,p,n] = 1  if l(a)==l(p) and l(a)!=l(n)
            mask[a,p,n] = -1 if l(a)!=l(p) and l(a)==l(n)
            mask[a,p,n] = 0  if l(a)==l(p) and l(a)==l(n)
                             or l(a)!=l(p) and l(a)!=l(n)
        """
        label_equal = tf.cast(
            tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)),
            dtype=tf.float32)
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        Rt = tf.math.subtract(i_equal_j, i_equal_k)

        return Rt

    def _get_pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = tf.linalg.diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.dtypes.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances