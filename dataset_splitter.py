import sys
import os
from collections import defaultdict
from typing import Dict, Union, Set

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(mol, include_chirality: bool = False) -> str:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols, use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(smiles):
    # Split
    train_size, val_size, test_size = 0.8, 0.1, 0.1
    id_data, val_ood, test_ood = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles([Chem.MolFromSmiles(_) for _ in smiles], use_indices=True)

    index_sets = sorted(list(scaffold_to_indices.values()), key=lambda x: len(x), reverse=True)

    for index_set in index_sets:
        if len(id_data) + len(index_set) <= train_size:
            id_data += index_set
            train_scaffold_count += 1
        elif len(val_ood) + len(index_set) <= val_size:
            val_ood += index_set
            val_scaffold_count += 1
        else:
            test_ood += index_set
            test_scaffold_count += 1

    # Map from indices to data
    id_data = [smiles[i] for i in id_data]
    val_ood = [smiles[i] for i in val_ood]
    test_ood = [smiles[i] for i in test_ood]

    # round 2
    train_size, val_size, test_size = 0.8, 0.1, 0.1
    train, val_id, test_id = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles([Chem.MolFromSmiles(_) for _ in id_data], use_indices=True)

    index_sets = sorted(list(scaffold_to_indices.values()), key=lambda x: len(x), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val_id) + len(index_set) <= val_size:
            val_id += index_set
            val_scaffold_count += 1
        else:
            test_id += index_set
            test_scaffold_count += 1

    # Map from indices to data
    train = [smiles[i] for i in train]
    val_id = [smiles[i] for i in val_id]
    test_id = [smiles[i] for i in test_id]

    return train, val_id, test_id, val_ood, test_ood


def size_split(smiles):
    train_size, val_size, test_size = 0.8, 0.1, 0.1
    id_data, val_ood, test_ood = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    sizes = [len(Chem.MolFromSmiles(_).GetAtoms()) for _ in smiles]
    sizes_dict = defaultdict(set)
    for i, size in enumerate(sizes):
        sizes_dict[size].add(i)

    index_sets = sorted(list(sizes_dict.values()), key=lambda x: len(x), reverse=True)

    for index_set in index_sets:
        if len(id_data) + len(index_set) <= train_size:
            id_data += index_set
            train_scaffold_count += 1
        elif len(val_ood) + len(index_set) <= val_size:
            val_ood += index_set
            val_scaffold_count += 1
        else:
            test_ood += index_set
            test_scaffold_count += 1

    # Map from indices to data
    id_data = [smiles[i] for i in id_data]
    val_ood = [smiles[i] for i in val_ood]
    test_ood = [smiles[i] for i in test_ood]

    # round 2
    train_size, val_size, test_size = 0.8, 0.1, 0.1
    train, val_id, test_id = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    sizes = [len(Chem.MolFromSmiles(_).GetAtoms()) for _ in smiles]
    sizes_dict = defaultdict(set)
    for i, size in enumerate(sizes):
        sizes_dict[size].add(i)

    index_sets = sorted(list(sizes_dict.values()), key=lambda x: len(x), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val_id) + len(index_set) <= val_size:
            val_id += index_set
            val_scaffold_count += 1
        else:
            test_id += index_set
            test_scaffold_count += 1

    # Map from indices to data
    train = [smiles[i] for i in train]
    val_id = [smiles[i] for i in val_id]
    test_id = [smiles[i] for i in test_id]

    return train, val_id, test_id, val_ood, test_ood


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