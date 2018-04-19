import numpy as np
import scipy.sparse as sp_sparse

from . import GeneExpressionDataset


class RetinaDataset(GeneExpressionDataset):
    def __init__(self, type='train'):

        self.save_path = 'data/'
        self.data_filename = 'retina_data_%s.npz' % type
        self.labels_filename = 'retina_labels_%s.dms' % type
        self.batches_filename = 'retina_batch_%s.dms' % type
        if type == 'train':
            cell_batches = np.reshape((np.loadtxt(self.save_path + self.batches_filename)-1), (19829, 1))
        if type == 'test':
            cell_batches = np.reshape((np.loadtxt(self.save_path + self.batches_filename)-1), (6610, 1))

        data = sp_sparse.load_npz(self.save_path + self.data_filename).toarray()
        data_with_batch_info = np.hstack((data, cell_batches))
        first_batch = data_with_batch_info[data_with_batch_info[:, -1] == 0.0]
        second_batch = data_with_batch_info[data_with_batch_info[:, -1] == 1.0]
        first_batch = first_batch[:, :-1]
        second_batch = second_batch[:, :-1]

        print("Finished preprocessing for Retina %s dataset" % type)
        super(RetinaDataset, self).__init__([sp_sparse.csr_matrix(first_batch), sp_sparse.csr_matrix(second_batch)],
                                            list_labels=[np.loadtxt(self.save_path + self.labels_filename)])
