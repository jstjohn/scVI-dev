from .brain_large import BrainLargeDataset
from .cortex import CortexDataset
from .dataset import GeneExpressionDataset
from .synthetic import SyntheticDataset
from .retina import RetinaDataset
from .cbmc import CbmcDataset

__all__ = ['SyntheticDataset',
           'CortexDataset',
           'BrainLargeDataset',
           'RetinaDataset',
           'GeneExpressionDataset',
           'CbmcDataset']


def load_datasets(dataset_name, unit_test=False):
    if dataset_name == 'synthetic':
        gene_dataset = SyntheticDataset()
    elif dataset_name == 'cortex':
        gene_dataset = CortexDataset()
    elif dataset_name == 'brain_large':
        gene_dataset = BrainLargeDataset(unit_test=unit_test)
    elif dataset_name == 'retina':
        gene_dataset = RetinaDataset(unit_test=unit_test)
    elif dataset_name == 'cbmc':
        gene_dataset = CbmcDataset(unit_test=unit_test)
    else:
        raise ValueError("No such dataset available")
    return gene_dataset
