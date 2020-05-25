from itertools import combinations, product

import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles, make_classification, make_gaussian_quantiles

def get_toy_dataset(shape: str, 
                    n_classes: int = 2, 
                    n_samples: int = 100, 
                    std: float = 0.05, 
                    seed = None):
    if shape=='moons':
        if n_classes > 2:
            print('Moons and Circles data only allow 2 classes, omitting additional requests.')
        return make_moons(n_samples=n_samples, noise=std, random_state=seed)
    
    elif shape=='circles':
        if n_classes > 2:
            print('Moons and Circles data only allow 2 classes, omitting additional requests.')
        return make_circles(n_samples=n_samples, noise=std, random_state=seed)
    
    elif shape=='normal':
        return make_classification(n_samples=n_samples,
                                   n_classes=n_classes,
                                   random_state=seed)
    
    elif shape=='quantiles':
        return make_gaussian_quantiles(n_samples=n_samples,
                                       n_classes=n_classes,
                                       random_state=seed)
    elif shape=='hierarchies':
        X1, y1 = make_moons(n_samples=n_samples//2, noise=std, random_state=seed)
        y1 += 10
        X2, y2 = make_moons(n_samples=n_samples//2, noise=std, random_state=seed)
        X2[:,0] += 2.5
        y2 += 20
        X = np.append(X1, X2, axis = 0)
        y = np.append(y1, y2)
        return X, y
    else:
        raise ValueError("Shape not found, must be either of 'moons', 'circles', 'normal', 'quantiles'")
        
def create_class_triplets(X, y, b:int=50):
    """
    Create triplets from a dataset X where the first two 
    entries stem from the same class, whereas the third
    entry from a different class.
    
    Note: be careful with large datasets (n > 300), since 
    the amount of possible triplets rises exponentially 
    and may quickly overload your computer's RAM. The 
    parameter b allows you to set an upper bound on how 
    many elements are used from each class to create 
    the triplets.
    
    Params
    ======
    X : the data set of shape (n, d) containing n d-dimen-
      sional data points
    y : the n class labels corresponding to the data X
    b : upper bound on how many elements should be 
      used from each class. Set to -1 to use all.
    
    Returns
    =======
    triplets : a numpy array containing all triplets      
    """
    triplets = []
    # for each class
    labels = np.unique(y)
    for label in labels:
        print('Label', label)
        elems = X[np.argwhere(y==label)][:,0,:][:b]
        # for all combinations of elems in that class
        for xi, xj in combinations(elems, 2):
            # combine with every element from other class(es)
            for cont_elem in X[np.argwhere(y!=label)][:,0,:]:
                triplets.append([xi, xj, cont_elem])
    
    print('Created triplets.')            
    # some additional processing
    print('Casting to numpy array ...')
    triplets = np.array(triplets)
    print('Shuffling ...')
    np.random.shuffle(triplets)
    print('Transposing axis 0 and 1 ...')
    triplets = np.transpose(triplets, (1,0,2))
    print('Done.')
    return triplets

def create_triplets_from_augmentations(X, aX, y):
    """
    Create triplets from a dataset X and its augmented
    verion aX. For each triplet, select one data point 
    from X as the first element, its augmented version 
    from aX as the second element, and one from a different
    class as third element. All elements in X, aX, y 
    have to be sorted in the same order.

    Params
    ======
    X : the data set of shape (n, d) containing n d-dimen-
      sional data points
    y : the n class labels corresponding to the data X
    
    Returns
    =======
    triplets : a numpy array containing all triplets      
    """
    triplets = []
    labels = np.unique(y)
    for label in labels:
        print('Label', label)
        label_indices = np.argwhere(y==label)
        for i in label_indices:
            org = X[i][0]
            aug = aX[i][0]
            for cont_elem in X[np.argwhere(y!=label)][:,0,:]:
                triplets.append([org, aug, cont_elem])
         
    print('Created triplets.')            
    # some additional processing
    print('Casting to numpy array ...')
    triplets = np.array(triplets)
    print('Shuffling ...')
    np.random.shuffle(triplets)
    print('Transposing axis 0 and 1 ...')
    triplets = np.transpose(triplets, (1,0,2))
    print('Done.')
    return triplets

def create_triplets_from_hierarchical_classes(X, y, b:int=50):
    """
    Create triplets from data X for which hierarchical
    class labels of the form 10, 11, 1x, ..., 20, 21, 2x, ...
    are provided in y.
    """
    triplets = []

    labels = np.unique(y)
    
    # for each label            
    for ref in labels:
        # for each other label in same top class 
        _same_top_class = labels//10==ref//10
        for pull in labels[_same_top_class]:
            if pull == ref:
                _push_labels = labels != ref # entirely diff class
            else:
                _push_labels = labels//10 != ref//10 # diff top class
            # for each label in other top class
            for push in labels[_push_labels]:
                print('Combination', ref, pull, push)
                # Get <b> random elements for each label
                ref_elems = X[np.argwhere(y==ref)][:,0,:]
                ref_elems = ref_elems[np.random.choice(len(ref_elems), b)]
                pull_elems = X[np.argwhere(y==pull)][:,0,:]
                pull_elems = pull_elems[np.random.choice(len(pull_elems), b)]
                push_elems = X[np.argwhere(y==push)][:,0,:]
                push_elems = push_elems[np.random.choice(len(push_elems), b)]
                # Add all element combinations
                for xi, xj, xk in product(ref_elems, pull_elems, push_elems):
                    triplets.append([xi, xj, xk])
       
    print('Created triplets.')            
    # some additional processing
    print('Casting to numpy array ...')
    triplets = np.array(triplets)
    print('Shuffling ...')
    np.random.shuffle(triplets)
    print('Transposing axis 0 and 1 ...')
    triplets = np.transpose(triplets, (1,0,2))
    print('Done.')
    return triplets
