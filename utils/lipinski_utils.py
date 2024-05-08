import typing as t
from pathlib import Path

import numpy as np
import rdkit.Chem.rdMolDescriptors as rdcmd
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Draw, Lipinski
from rdkit.Chem.rdmolfiles import MolFromSmiles

from .mol_methods import verify_sequence

from .filter import apply_filters


class SmilesError(Exception):
    pass


def is_valid(smiles: str) -> bool:
    """Returns True if smiles encoding represents a valid compound."""
    return MolFromSmiles(smiles) is not None


def log_partition_coefficient(smiles):
    """
    Returns the octanol-water partition coefficient given a molecule SMILES
    string
    """
    try:
        mol = MolFromSmiles(smiles)
    except Exception as e:
        raise SmilesError("%s returns a None molecule" % smiles)

    return Crippen.MolLogP(mol)


def lipinski_hard_filter(
    smiles_list: t.List[str],
    max_n_hdonors: int = 5,
    max_n_hacceptors: int = 10,
    max_mol_weight: float = 500.0,
    min_mol_weight: float = 300.0,
    max_mol_logp: int = 5,
) -> t.List[str]:
    """Given a list of SMILES strings returns a list of those SMILES
    that are valid and satisfy the Lipinski's five rules.
    """
    filtered_smiles = list()
    for smile in smiles_list:
        mol = MolFromSmiles(smile)
        if mol is None:
            continue

        num_hdonors = Lipinski.NumHDonors(mol)
        num_hacceptors = Lipinski.NumHAcceptors(mol)
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Crippen.MolLogP(mol)
        ring_count = Lipinski.RingCount(mol)
        heavy_atom = Lipinski.HeavyAtomCount(mol)

        if num_hdonors >= max_n_hdonors:
            continue
        if num_hacceptors >= max_n_hacceptors:
            continue
        if mol_weight >= max_mol_weight:
            continue
        if mol_weight <= min_mol_weight:
            continue
        if mol_logp >= max_mol_logp:
            continue
        if apply_filters(smile):
            filtered_smiles.append(smile)
        else:
            continue

    return filtered_smiles


def lipinski_filter(
    smiles_list: t.List[str],
    max_n_hdonors: int = 5,
    max_n_hacceptors: int = 10,
    max_mol_weight: float = 500.0,
    min_mol_weight: float = 300.0,
    max_mol_logp: int = 5,
) -> t.List[str]:
    """Given a list of SMILES strings returns a list of those SMILES
    that are valid and satisfy the Lipinski's five rules.
    """
    filtered_smiles = list()
    for smile in smiles_list:
        mol = MolFromSmiles(smile)
        if mol is None:
            continue

        num_hdonors = Lipinski.NumHDonors(mol)
        num_hacceptors = Lipinski.NumHAcceptors(mol)
        mol_weight = Descriptors.MolWt(mol)
        mol_logp = Crippen.MolLogP(mol)
        ring_count = Lipinski.RingCount(mol)
        heavy_atom = Lipinski.HeavyAtomCount(mol)

        if num_hdonors >= max_n_hdonors:
            continue
        if num_hacceptors >= max_n_hacceptors:
            continue
        if mol_weight >= max_mol_weight:
            continue
        if mol_weight <= min_mol_weight:
            continue
        if mol_logp >= max_mol_logp:
            continue
        filtered_smiles.append(smile)

    return filtered_smiles


def compute_lipinski(
    smiles: str,
    num_hdonors_ref: int = 5,
    num_hacceptors_ref: int = 10,
    mol_weight_ref: int = 500,
    mol_logp_ref: int = 5,
):
    """
    Returns which of Lipinski's rules a molecule has failed, or an empty list
    https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five
    Lipinski's rules are:
    Hydrogen bond donors <= 5
    Hydrogen bond acceptors <= 10
    Molecular weight < 500 daltons
    logP < 5
    """
    passed = []
    failed = []

    mol = MolFromSmiles(smiles)
    if mol is None:
        raise Exception("%s is not a valid SMILES string" % smiles)

    num_hdonors = Lipinski.NumHDonors(mol)
    num_hacceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Crippen.MolLogP(mol)
    ring_count = Lipinski.RingCount(mol)
    heavy_atom = Lipinski.HeavyAtomCount(mol)
    rob = rdcmd.CalcNumRotatableBonds(mol)
    failed = []

    lipinski_score_array = []

    lipinski_score = {}

    if num_hdonors > num_hdonors_ref:
        failed.append("Over 5 H-bond donors, found %s" % num_hdonors)
        binary = 0
    else:
        passed.append("Found %s H-bond donors" % num_hdonors)
        binary = 1

    lipinski_score_array.append(binary)
    lipinski_score["n_hdonors"] = num_hdonors
    if num_hacceptors > num_hacceptors_ref:
        failed.append("Over 10 H-bond acceptors, found %s" % num_hacceptors)
        binary = 0
    else:
        passed.append("Found %s H-bond acceptors" % num_hacceptors)
        binary = 1
    lipinski_score_array.append(binary)
    lipinski_score["n_hacceptors"] = num_hacceptors
    if mol_weight >= mol_weight_ref:
        failed.append("Molecular weight over 500, calculated %s" % mol_weight)
        binary = 0
    else:
        passed.append("Molecular weight: %s" % mol_weight)
        binary = 1

    lipinski_score["mol_weight"] = mol_weight
    lipinski_score_array.append(binary)
    if mol_logp >= mol_logp_ref:
        failed.append("Log partition coefficient over 5, calculated %s" % mol_logp)
        binary = 0
    else:
        passed.append("Log partition coefficient: %s" % mol_logp)
        binary = 1
    lipinski_score_array.append(binary)
    if rob < 10:
        passed.append("RotatableBonds value is: %s" % mol_logp)
        binary = 1
    else:
        failed.append("RotatableBonds value is over 10, value: %s" % mol_logp)
        binary = 0

    lipinski_score_array.append(binary)

    lipinski_score["mol_logp"] = mol_logp
    # addition to the lipinski scores
    lipinski_score["ring_count"] = ring_count
    lipinski_score["heavy_atom"] = heavy_atom

    return (passed, failed, np.asanyarray(lipinski_score_array), lipinski_score)


def lipinski_pass(smiles):
    """
    Wraps around lipinski trial, but returns a simple pass/fail True/False
    """
    passed, failed, _, _ = lipinski_trial(smiles)
    if failed:
        return False
    else:
        return True


def compute_logp(smi: str) -> float:
    """User-defined function that takes in individual smiles
    and outputs a fitness value.
    """
    # logP fitness
    return Descriptors.MolLogP(Chem.MolFromSmiles(smi))


def compute_qed(smi: str):
    # QED
    return QED.qed(Chem.MolFromSmiles(smi))


def draw_compounds(smiles: str, path: str = "", file_name: str = "smiles.png"):
    Path(path).mkdir(parents=True, exist_ok=True)
    mols = [Chem.MolFromSmiles(smi_) for smi_ in smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=len(mols), returnPNG=False)
    img.save(f"{path}/{file_name}")
