import gzip
import numpy as np


def export_cbmc_unit_test(cbmc_dataset, n_cells=50, n_genes=10):
    cbmc_dataset.subsample_cells(n_cells)
    cbmc_dataset.subsample_genes(n_genes)
    subsampled_data = np.concatenate((cbmc_dataset.gene_names.reshape(1, -1), cbmc_dataset.X)).T

    with gzip.open('tests/data/cbmc_subsampled.csv.gz', 'w') as gzipfile:
        row = subsampled_data[0]  # Dummy row
        line = ','.join(row) + '\n'
        gzipfile.write(line.encode())
        for row in subsampled_data:
            line = ','.join(row) + '\n'
            gzipfile.write(line.encode())
