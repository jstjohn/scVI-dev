import numpy as np

import csv
import tarfile
from pathlib import Path
from scipy import io

from .dataset import GeneExpressionDataset


class BrainSmallDataset(GeneExpressionDataset):
    url = "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/neuron_9k/" + \
          "neuron_9k_filtered_gene_bc_matrices.tar.gz"

    # TODO: Create test file and add case when unit_test is True
    # TODO: Should I only keep one of the 'filtered_gene_bc_matrices.tar.gz' and 'filtered_gene_bc_matrices'? (ask after submitting pull request)
    def __init__(self, unit_test=False):
        self.save_path = 'data/'
        self.unit_test = unit_test

        self.download_name = 'filtered_gene_bc_matrices.tar.gz'
        self.dir_name = self.download_name[:-7] + "/mm10/"

        expression_data, gene_names = self.download_and_preprocess()

        super(BrainSmallDataset, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                expression_data), gene_names=gene_names)

    def preprocess(self):
        if not Path(self.save_path + self.dir_name).is_dir():
            print("Unzip Brain Small data")
            tar = tarfile.open(self.save_path + self.download_name)
            tar.extractall(self.save_path)
            tar.close()

        print("Preprocessing Brain Small data")
        gene_names = []
        barcodes = []

        def _store_tsv_data(filename, des):
            print("Loading " + filename)
            with open(self.save_path + self.dir_name + filename, "r") as tsv:
                data_reader = csv.reader(tsv, delimiter="\t", quotechar='"')
                for row in data_reader:
                    des.append(row[0])

        _store_tsv_data("genes.tsv", gene_names)
        _store_tsv_data("barcodes.tsv", barcodes)

        print("Loading expression data")
        expression_data = io.mmread(self.save_path + self.dir_name + "matrix.mtx").T.A
        gene_names = np.array(gene_names, dtype=np.str)

        selected = np.std(expression_data, axis=0).argsort()[-3000:][::-1]
        expression_data = expression_data[:, selected]
        gene_names = gene_names[selected]

        return expression_data, gene_names
