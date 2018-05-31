import numpy as np

import csv
import tarfile
import os
from pathlib import Path
from scipy import io, sparse

from .dataset import GeneExpressionDataset


class BrainSmallDataset(GeneExpressionDataset):
    url = "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/neuron_9k/" + \
          "neuron_9k_filtered_gene_bc_matrices.tar.gz"

    # TODO: Should I only keep one of the 'filtered_gene_bc_matrices.tar.gz' and 'filtered_gene_bc_matrices'? (ask after submitting pull request)
    def __init__(self, unit_test=False):
        self.save_path = 'data/'
        self.unit_test = unit_test

        self.download_name = 'filtered_gene_bc_matrices.tar.gz'
        if not unit_test:
            self.gene_file = "filtered_gene_bc_matrices/mm10/genes.tsv"
            self.expression_file = "filtered_gene_bc_matrices/mm10/matrix.mtx"
        else:
            self.gene_file = "../tests/data/brain_small_subsampled/mm10/genes_subsampled.tsv"
            self.expression_file = "../tests/data/brain_small_subsampled/mm10/matrix_subsampled.mtx"

        expression_data, gene_names = self.download_and_preprocess()

        super(BrainSmallDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                expression_data), gene_names=gene_names)

    def export_unit_test(self, n_cells=50, n_genes=10):
        self.subsample_cells(n_cells)
        self.subsample_genes(n_genes)
        matrix = sparse.coo_matrix(self.X.T)

        path = "tests/data/brain_small_subsampled/mm10/"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "genes_subsampled.tsv", "w") as tsv:
            for row in self.gene_names:
                tsv.write(row + "\n")

        io.mmwrite(path + "matrix_subsampled.mtx", matrix)

    def preprocess(self):
        if not Path(self.save_path + self.download_name[:-7]).is_dir():
            print("Unzipping Brain Small data")
            tar = tarfile.open(self.save_path + self.download_name)
            tar.extractall(self.save_path)
            tar.close()

        print("Preprocessing Brain Small data")
        gene_names = []

        def _store_tsv_data(file, des):
            with open(file, "r") as tsv:
                data_reader = csv.reader(tsv, delimiter="\t", quotechar='"')
                for row in data_reader:
                    des.append(row[0])

        _store_tsv_data(self.save_path + self.gene_file, gene_names)
        expression_data = io.mmread(self.save_path + self.expression_file).T.A

        gene_names = np.array(gene_names, dtype=np.str)

        selected = np.std(expression_data, axis=0).argsort()[-3000:][::-1]
        expression_data = expression_data[:, selected]
        gene_names = gene_names[selected]

        return expression_data, gene_names
