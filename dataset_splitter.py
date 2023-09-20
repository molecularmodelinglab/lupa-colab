import sys
import os

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


# TODO
def scaffold_split(smiles):
    raise NotImplemented


# TODO
def size_split(smiles):
    raise NotImplemented


# TODO
def assay_split(smiles):
    raise NotImplemented


# TODO
def smi2fp(smi, radius: int = 3, n_bits: int = 1024, pca: int = None):
    raise NotImplemented


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="the path to a csv file contianing smiles and labels")
    parser.add_argument("-o", "--output", type=str, required=False, default=os.path.join(os.getcwd(), "splits"),
                        help="the output dir path for generated splits and datasets; default to 'splits' in cwd")
    parser.add_argument("-s", "--smi-col", type=str, required=False, default="SMILES",
                        help="name of the column containing the SMILES strings; default is 'SMILES'")
    parser.add_argument("-l", "--label-col", type=str, required=False, default="Label",
                        help="name of the column containing the labels; default is 'Label'")
    parser.add_argument("-m", "--method", type=str, required=False, default="scaffold",
                        help="which domain split to use: 'scaffold', 'size' or 'assay'; default is 'scaffold'")
    parser.add_argument("-r", "--radius", type=int, required=False, default=3,
                        help="radius of ecfp fingerprints to use (default is 3, changes how many bits are on)")
    parser.add_argument("-b", "--nbits", type=int, required=False, default=1024,
                        help="dimension of the fingerprint")
    parser.add_argument("-p", "--pca", type=int, required=False, default=None,
                        help="dimension to reduce fp to using pca; if None will not use pca (default is None)")