# reduce the dimensionality of the embeddings
import umap
import torch
import numpy as np
from sklearn.manifold import TSNE


class UMAPReducer:
    def __init__(self, n_components=2, **kwargs):
        self.reducer = umap.UMAP(n_components=n_components, **kwargs)

    def fit_transform(self, tensor):
        return self.reducer.fit_transform(tensor)
    

class TSNEReducer:
    def __init__(self, n_components=2):
        self.reducer = TSNE(n_components=n_components)

    def fit_transform(self, tensor):
        return self.reducer.fit_transform(tensor)


def recude_dim(embeddings, n_components=2,method='umap'):
    """
    :param embeddings: the embeddings of the nodes
    :param n_components: the number of components to reduce to
    :return: the embeddings reduced to n_components
    """
    if method == 'umap':
        reducer = UMAPReducer(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNEReducer(n_components=n_components)
    else:
        raise ValueError('Invalid method for dimensionality reduction')
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(reduced_embeddings.shape) 
    return reduced_embeddings
