import pandas as pd
from sklearn.decomposition import PCA
import warnings
class GenomeTag:
    def __init__(self, itemDataset, tagDataset, tagScoreDataset):
        if type(itemDataset) != pd.core.frame.DataFrame:
            raise TypeError('Cannot recognize the item dataset')
        elif type(tagDataset) != pd.core.frame.DataFrame:
            raise TypeError('Cannot recognize the tag dataset')
        elif type(tagScoreDataset) != pd.core.frame.DataFrame:
            raise TypeError('Cannot recognize the tag weight dataset')
        else:
            self.itemData = itemDataset.iloc[:, :].values
            self.tagData = tagDataset.iloc[:, :].values
            self.tagScoreData = tagScoreDataset.iloc[:, :].values

    def applyPCA(self, cutoff = 0.5):
        pca = PCA(n_components = cutoff/100, svd_solver = 'full')
        self.tagScoreData = pca.fit_transform(self.tagScoreData)

    def process(self):
        message = 'Assume each row either in the format of \n\tmovieId weight_of_each_tag\n\tweight_of_each_tag\nWe ' \
                  'need to take only the matrix of weight.'
        warnings.warn(message, SyntaxWarning, stacklevel = 0)
        n = self.itemData.shape[0]
        m = self.tagData.shape[0]
        if self.tagScoreData.shape[0] < n:
            message = 'There will be some items that have not had any tags yet.'
            warnings.warn(message, RuntimeWarning, stacklevel = 0)
        elif self.tagScoreData.shape[0] > n:
            raise IndexError('X - dimension out of bound.')

        if self.tagScoreData.shape[1] == m:
            pass
        elif self.tagScoreData.shape[1] == m + 1:
            retain = list(range(1, m + 1))
            self.tagScoreData = self.tagScoreData[:, retain]
        else:
            raise IndexError('Y - dimension out of bound.')

    #def _calGuassian_(self, a, b):


    #def _buildDistancematrix_(self, metric = 'cos'):
    #    if metric == 'guassian':
