====
scVI
====

.. image:: https://travis-ci.org/YosefLab/scVI-dev.svg?branch=master
    :target: https://travis-ci.org/YosefLab/scVI-dev

.. image:: https://codecov.io/gh/YosefLab/scVI-dev/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/YosefLab/scVI-dev

.. image:: https://readthedocs.org/projects/scvi/badge/?version=latest
        :target: https://scvi.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Single-cell Variational Inference


* Free software: MIT license
* Documentation: https://scvi.readthedocs.io.


Installation
--------

1. Install PyTorch: http://pytorch.org

2. Install this repository from source: ``git clone https://github.com/YosefLab/scVI-dev.git``

3. Refer to the `examples/` subdirectory for Jupyter notebooks that illustrate usage. 


Benchmarks
--------

To recreate the results appearing in the paper referenced below, run

.. code-block::

    python ./run_benchmarks.py --dataset=cortex 

References
--------

Romain Lopez, Jeffrey Regier, Michael B Cole, Michael Jordan, Nir Yosef.
**"Bayesian Inference for a Generative Model of Transcriptome Profiles from Single-cell RNA Sequencing."**
In submission. Preprint available at https://www.biorxiv.org/content/early/2018/03/30/292037
