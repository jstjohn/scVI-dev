#!/usr/bin/env python

"""Run all the benchmarks with specific parameters"""
import argparse

from scvi.models import VAEQC, VAE, VAEC, VAECQC, SVAEC, SVAECQC
from scvi.benchmark import run_benchmarks

models = {
    "VAEQC": VAEQC,
    "VAECQC": VAECQC,
    "VAE": VAE,
    "VAEC": VAEC,
    "SVAEC": SVAEC,
    "SVAECQC": SVAECQC,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250, help="how many times to process the dataset")
    parser.add_argument("--dataset", type=str, default="cortex", help="which dataset to process")
    parser.add_argument("--train-labels", action='store_true', help="Do you want to train on labels as well as data?")
    parser.add_argument("--nobatches", action='store_true', help="whether to ignore batches")
    parser.add_argument("--model", default="VAE", choices=list(models.keys()),
                        help="Which model would you like to run? (default %(default)s)")
    parser.add_argument("--nocuda", action='store_true',
                        help="whether to use cuda (will apply only if cuda is available")
    parser.add_argument("--benchmark", action='store_true',
                        help="whether to use cuda (will apply only if cuda is available")
    args = parser.parse_args()

    run_benchmarks(args.dataset, model=models[args.model], n_epochs=args.epochs, use_batches=(not args.nobatches),
                   use_cuda=(not args.nocuda), show_batch_mixing=True, benchmark=args.benchmark,
                   train_labels=args.train_labels)
