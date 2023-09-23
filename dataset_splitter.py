import os
import random
from collections import defaultdict, Counter
from typing import Dict, Union, Set

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.decomposition import PCA


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


def size_to_smiles(mols):
    sizes = defaultdict(set)
    for i, mol in enumerate(mols):
        _size = len(mol.GetAtoms())
        sizes[_size].add(i)
    return sizes


def scaffold_split(smiles):
    # Split
    train_size, val_size, test_size = 0.8 * len(smiles), 0.1 * len(smiles), 0.1 * len(smiles)
    id_data, val_ood, test_ood = [], [], []
    val_ood_membership, test_ood_membership = [], []
    val_ood_distance, test_ood_distance = [], []

    id_data_domain_ids, val_ood_domain_ids, test_ood_domain_ids = [], [], []
    id_data_domain_count, val_ood_domain_count, test_ood_domain_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles([Chem.MolFromSmiles(_) for _ in smiles], use_indices=True)
    scaffold_to_indices = {key: (val, len(Chem.MolFromSmiles(key).GetAtoms())) for key, val in
                           scaffold_to_indices.items()}

    _sizes = [_[1] for _ in scaffold_to_indices.values()]
    size_counts = Counter(_sizes)
    size_ranks = sorted([(key, val) for key, val in size_counts.items()], key=lambda x: x[1], reverse=True)

    _scaffold_to_indices = {}
    for (key, _) in size_ranks:
        for _key, _val in scaffold_to_indices.items():
            if _val[1] == key:
                _scaffold_to_indices[_key] = _val
    scaffold_to_indices = _scaffold_to_indices

    domain_sizes = {}

    _id_size_set = set()
    for i, (_scaf, (index_set, _size)) in enumerate(scaffold_to_indices.items()):
        domain_sizes[i] = _size
        if len(id_data) + len(index_set) <= train_size:
            _id_size_set.add(_size)
            id_data += index_set
            id_data_domain_ids.append(_scaf)
            id_data_domain_count += 1
        elif len(val_ood) + len(index_set) <= val_size:
            val_ood += index_set
            val_ood_membership += [i for _ in range(len(index_set))]
            _dist = np.min([np.abs(_size - _) for _ in _id_size_set])
            val_ood_distance += [_dist for _ in range(len(index_set))]
            val_ood_domain_ids.append(i)
            val_ood_domain_count += 1
        else:
            test_ood += index_set
            test_ood_membership += [i for _ in range(len(index_set))]
            _dist = np.min([np.abs(_size - _) for _ in _id_size_set])
            test_ood_distance += [_dist for _ in range(len(index_set))]
            test_ood_domain_ids.append(i)
            test_ood_domain_count += 1

    # round 2
    train_size, val_size, test_size = 0.8 * len(id_data), 0.1 * len(id_data), 0.1 * len(id_data)
    train, val_id, test_id = [], [], []
    train_id_domain_count, val_id_domain_count, test_id_domain_count = 0, 0, 0
    train_domain_ids, val_id_domain_ids, test_id_domain_ids = [], [], []

    id_scaffold_indices = {i: val for i, (key, val) in enumerate(scaffold_to_indices.items()) if
                           key in id_data_domain_ids}
    l = list(id_scaffold_indices.items())
    random.shuffle(l)
    id_scaffold_indices = dict(l)

    for i, (index_set, _size) in id_scaffold_indices.items():
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_domain_ids.append(i)
            train_id_domain_count += 1
        elif len(val_id) + len(index_set) <= val_size:
            val_id += index_set
            val_id_domain_ids.append(i)
            val_id_domain_count += 1
        else:
            test_id += index_set
            test_id_domain_ids.append(i)
            test_id_domain_count += 1

    ood_val_df = pd.DataFrame(
        {"domain_id": val_ood_membership, "ood_dist": val_ood_distance})
    ood_test_df = pd.DataFrame(
        {"domain_id": test_ood_membership, "ood_dist": test_ood_distance})

    return train, val_id, test_id, val_ood, test_ood, ood_val_df, ood_test_df


def size_split(smiles):
    # Split
    train_size, val_size, test_size = 0.8 * len(smiles), 0.1 * len(smiles), 0.1 * len(smiles)
    id_data, val_ood, test_ood = [], [], []
    val_ood_membership, test_ood_membership = [], []
    val_ood_distance, test_ood_distance = [], []

    id_data_domain_ids, val_ood_domain_ids, test_ood_domain_ids = [], [], []
    id_data_domain_count, val_ood_domain_count, test_ood_domain_count = 0, 0, 0

    # Map from scaffold to index in the data
    size_to_indices = size_to_smiles([Chem.MolFromSmiles(_) for _ in smiles], use_indices=True)
    size_to_indices = {key: (val, len(Chem.MolFromSmiles(key).GetAtoms())) for key, val in
                       size_to_indices.items()}

    l = sorted(list(size_to_indices.items()), key=lambda x: x[-1][-1])
    size_to_indices = dict(l)
    domain_sizes = {}

    _max_id_size = 0
    for i, (_scaf, (index_set, _size)) in enumerate(size_to_indices.items()):
        domain_sizes[i] = _size
        if len(id_data) + len(index_set) <= train_size:
            if _size > _max_id_size:
                _max_id_size = _size
            id_data += index_set
            id_data_domain_ids.append(_scaf)
            id_data_domain_count += 1
        elif len(val_ood) + len(index_set) <= val_size:
            val_ood += index_set
            val_ood_membership += [i for _ in range(len(index_set))]
            _dist = _size - _max_id_size
            val_ood_distance += [_dist for _ in range(len(index_set))]
            val_ood_domain_ids.append(i)
            val_ood_domain_count += 1
        else:
            test_ood += index_set
            val_ood_membership += [i for _ in range(len(index_set))]
            _dist = _size - _max_id_size
            val_ood_distance += [_dist for _ in range(len(index_set))]
            test_ood_domain_ids.append(i)
            test_ood_domain_count += 1

    # round 2
    train, val_id, test_id = [], [], []
    train_id_domain_count, val_id_domain_count, test_id_domain_count = 0, 0, 0
    train_domain_ids, val_id_domain_ids, test_id_domain_ids = [], [], []

    id_size_indices = {i: val for i, (key, val) in enumerate(size_to_indices.items()) if
                           key in id_data_domain_ids}
    l = list(id_size_indices.items())
    random.shuffle(l)
    id_size_indices = dict(l)

    for i, (index_set, _size) in id_size_indices.items():
        if len(id_data) + len(index_set) <= train_size:
            train += index_set
            train_domain_ids.append(i)
            train_id_domain_count += 1
        elif len(val_ood) + len(index_set) <= val_size:
            val_id += index_set
            val_id_domain_ids.append(i)
            val_id_domain_count += 1
        else:
            test_id += index_set
            test_id_domain_ids.append(i)
            test_id_domain_count += 1

    # Map from indices to data
    # train = [smiles[i] for i in train]
    # val_ood = [smiles[i] for i in val_ood]
    # test_ood = [smiles[i] for i in test_ood]
    # val_id = [smiles[i] for i in val_id]
    # test_id = [smiles[i] for i in test_id]

    ood_val_df = pd.DataFrame(
        {"domain_id": val_ood_membership, "ood_dist": val_ood_distance})
    ood_test_df = pd.DataFrame(
        {"domain_id": test_ood_membership, "ood_dist": test_ood_distance})

    return train, val_id, test_id, val_ood, test_ood, ood_val_df, ood_test_df


# TODO
def assay_split(smiles):
    raise NotImplemented


def smi2fp(smi, radius: int = 3, n_bits: int = 1024):
    return np.array(list(AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(smi), radius=radius, nBits=n_bits)))


if __name__ == '__main__':
    import argparse
    import warnings

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="the path to a csv file contianing smiles and labels")
    parser.add_argument("-o", "--output", type=str, required=False, default=os.path.join(os.getcwd(), "splits"),
                        help="the output dir path for generated splits and datasets; default to 'splits' in cwd")
    parser.add_argument("-s", "--smi_col", type=str, required=False, default="SMILES",
                        help="name of the column containing the SMILES strings; default is 'SMILES'")
    parser.add_argument("-l", "--label_col", type=str, required=False, default="Label",
                        help="name of the column containing the labels; default is 'Label'")
    parser.add_argument("-m", "--method", type=str, required=False, default="scaffold",
                        help="which domain split to use: 'scaffold', 'size' or 'assay'; default is 'scaffold'")
    parser.add_argument("-r", "--radius", type=int, required=False, default=3,
                        help="radius of ecfp fingerprints to use (default is 3, changes how many bits are on)")
    parser.add_argument("-b", "--nbits", type=int, required=False, default=1024,
                        help="dimension of the fingerprint")
    parser.add_argument("-p", "--pca", type=int, required=False, default=None,
                        help="dimension to reduce fp to using pca; if None will not use pca (default is None)")

    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = data[[args.smi_col, args.label_col]]
    data.dropna(inplace=True)
    data["ROMol"] = data[args.smi_col].apply(Chem.MolFromSmiles)

    _tmp = data.dropna()
    if len(data) != len(_tmp):
        warnings.warn(f"detected {len(data) - len(_tmp)} datapoint with NaN, these are automatically removed")
    data = _tmp.copy()
    del _tmp

    if args.method == "scaffold":
        train, val_id, test_id, val_ood, test_ood, ood_val_df, ood_test_df = scaffold_split(data[args.smi_col].to_list())
    elif args.method == "size":
        train, val_id, test_id, val_ood, test_ood, ood_val_df, ood_test_df = size_split(data[args.smi_col].to_list())
    else:
        raise NotImplementedError("Sorry James only has 'scaffold' and 'size' right now")

    os.makedirs(args.output, exist_ok=True)

    del data["ROMol"]

    _pca = PCA(n_components=int(args.pca)) if args.pca is not None else None
    for _name, idxs in [("train", train), ("val_id", val_id), ("test_id", test_id), ("val_ood", val_ood), ("test_ood", test_ood)]:
        _df = data.iloc[idxs].copy().reset_index(drop=True)
        _fps = np.array([smi2fp(_) for _ in _df[args.smi_col]])
        if _pca:
            if _name == "train":
                _pca.fit(_fps)
            _fps = _pca.transform(_fps)
        fps = pd.DataFrame(_fps).reset_index(drop=True)
        fps.columns = [f"fp_{i+1}" for i in range(len(fps.columns))]
        if _name == "val_ood":
            _df = pd.concat((_df, ood_val_df), axis=1).reset_index(drop=True)
        if _name == "test_ood":
            _df = pd.concat((_df, ood_test_df), axis=1).reset_index(drop=True)
        _df = pd.concat((_df, fps), axis=1).reset_index(drop=True)
        _df.to_csv(os.path.join(args.output, f"{_name}.csv"), index=False)

    print(f"wrote files to {args.output}")
