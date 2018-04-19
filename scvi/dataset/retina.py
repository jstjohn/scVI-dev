import numpy as np
import scipy.sparse as sp_sparse

from . import GeneExpressionDataset


class RetinaDatasetTrain(GeneExpressionDataset):
    def __init__(self):
        self.save_path = 'data/'
        self.data_filename = 'retina_data_train.npz'
        self.labels_filename = 'retina_labels_train.dms'
        super(RetinaDatasetTrain, self).__init__([sp_sparse.csr_matrix(sp_sparse.load_npz(self.save_path
                                                                                          + self.data_filename).A)],
                                                 list_labels=[np.loadtxt(self.save_path + self.labels_filename)])


class RetinaDatasetTest(GeneExpressionDataset):
    def __init__(self):
        self.save_path = 'data/'
        self.data_filename = 'retina_data_test.npz'
        self.labels_filename = 'retina_labels_test.dms'
        super(RetinaDatasetTest, self).__init__([sp_sparse.csr_matrix(sp_sparse.load_npz(self.save_path
                                                                                         + self.data_filename).A)],
                                                list_labels=np.loadtxt(self.save_path + self.labels_filename))
