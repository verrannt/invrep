import numpy as np

def add_random_noise(X, std=0.1):
    """
    Augment data values by adding random noise in each
    dimension.
    This operation is in-place for numpy-arrays, i.e. if 
    you don't want the array it to change pass X.copy()
    as argument when using numpy-arrays.
    
    Params
    ======
    X : the array of data points to use. It should be of 
      shape (n, d) where n is the number of data points
      and d the dimension of each data point.
    std : the standard-deviation of the noise to be
      applied, drawn from a normal distribution.
      
    Returns
    =======
    X : the augmented array.    
    """
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] += np.random.normal(loc=0., scale=std)
    return X