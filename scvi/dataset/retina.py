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
            cell_batches = np.reshape((np.loadtxt(self.save_path + self.batches_filename) - 1), (19829, 1))
            labels = np.reshape((np.loadtxt(self.save_path + self.labels_filename)), (19829, 1))
        if type == 'test':
            cell_batches = np.reshape((np.loadtxt(self.save_path + self.batches_filename) - 1), (6610, 1))
            labels = np.reshape((np.loadtxt(self.save_path + self.labels_filename)), (6610, 1))

        data = sp_sparse.load_npz(self.save_path + self.data_filename).toarray()
        print(data.shape)
        data_with_info = np.hstack((data, labels, cell_batches))
        print(data_with_info.shape)
        first_batch = data_with_info[data_with_info[:, -1] == 0.0]
        second_batch = data_with_info[data_with_info[:, -1] == 1.0]
        first_batch = first_batch[:, :-1]
        second_batch = second_batch[:, :-1]
        print(first_batch[:, :-1].shape)

        print("Finished preprocessing for Retina %s dataset" % type)
        # super(RetinaDataset, self).__init__([sp_sparse.csr_matrix(data)],
        #                                     list_labels=[labels])
        super(RetinaDataset, self).__init__([sp_sparse.csr_matrix(first_batch[:, :-1]),
                                             sp_sparse.csr_matrix(second_batch[:, :-1])],
                                            list_labels=[first_batch[:, -1], second_batch[:, -1]])
