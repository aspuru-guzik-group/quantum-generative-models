import os
import sys
import typing as t
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import selfies as sf
import torch
from orquestra.qml.api import GenerativeModel, convert_to_numpy
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi

from utils.lipinski_utils import (
    compute_lipinski,
    compute_logp,
    compute_qed,
    draw_compounds,
)

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def decode_smiles(path: str):
    (
        encoding_list,
        encoding_alphabet,
        largest_molecule_len,
        encoding_list_smiles,
        encoding_alphabet_smiles,
        largest_molecule_len_smiles,
    ) = get_selfie_and_smiles_encodings_for_dataset(path)
    symbol_to_int = dict((c, i) for i, c in enumerate(encoding_alphabet))
    all_selfies_hot_encoding = []
    all_selfies_hot_encoding_list = []
    i = 0
    for selfies_ in encoding_list:
        selfie = selfies_
        selfie += "[nop]" * (largest_molecule_len - sf.len_selfies(selfies_))
        selfies_list = sf.split_selfies(selfie)
        # print(sf.len_selfies(selfie))
        selfies_hot_encoding = sf.batch_selfies_to_flat_hot(selfies_list, symbol_to_int)
        not_ok = True
        for j in selfies_hot_encoding:
            if len(j) == len(encoding_alphabet):
                """"""
            else:
                not_ok = False
                print(f"{i}-KO!")

        if not_ok:
            all_selfies_hot_encoding.append(np.asarray(selfies_hot_encoding))
            all_selfies_hot_encoding_list.append(selfies_hot_encoding)
        i += 1
    return (
        torch.tensor(all_selfies_hot_encoding_list),
        encoding_list,
        encoding_alphabet,
        largest_molecule_len,
        symbol_to_int,
    )


def get_selfie_and_smiles_encodings_for_dataset(file_path: str):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = pd.read_csv(file_path).dropna()

    smiles_list = np.asanyarray(df.smiles)
    smiles_alphabet = list(set("".join(smiles_list)))
    smiles_alphabet.append(" ")  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    print("--> Translating SMILES to SELFIES...")
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add("[nop]")
    selfies_alphabet = list(all_selfies_symbols)

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print("Finished translating SMILES to SELFIES.")

    return (
        selfies_list,
        selfies_alphabet,
        largest_selfies_len,
        smiles_list,
        smiles_alphabet,
        largest_smiles_len,
    )


def truncate_smiles(
    smiles: t.Iterable[str], padding_char: str = "_", min_length: int = 1
) -> t.List[str]:
    """Truncates a list of SMILES such that only characters prior to the first
    occurrence of <padding_char> remain. If the length of the truncated SMILE is less than the <min_length>
    parameter, that smile will be removed from the list. Performs the operation inplace.

    Examples::
        >>> smiles = ['cc2_N', 'cc2__', 'cc2']
        >>> truncate_smiles(smiles)
        >>> print(smiles)
        ['cc2', 'cc2', 'cc2']

    Args:
        smiles (t.List[str]): list of SMILES strings.
        padding_char (str, optional): representation of the padding character. Defaults to "_".
    """

    # for faster appends if list of SMILES is very long
    truncated_smiles = list()

    for smile in smiles:
        # .index(s) raises an error if substring is not found
        try:
            truncated_smile = smile[: smile.index(padding_char)]
        except ValueError:
            truncated_smile = smile

        if len(truncated_smile) >= min_length:
            truncated_smiles.append(truncated_smile)

    return list(truncated_smiles)


def generate_bulk_samples(
    model: GenerativeModel,
    n_samples: int,
    n_samples_per_attempt: int,
    max_attempts: int = 10,
    *,
    unique: bool = True,
    prior: Optional[GenerativeModel] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Use this function when generating high number of samples.

    Args:
        model (GenerativeModel): model to generate samples with.
        n_samples (int): total number of samples to generate.
        n_samples_per_attempt(int): number of samples to generate at each pass.
        max_attempts (int): maximum number of attempts to generate the specified number of samples, before giving up. Defaults to 10.
        unique (bool): whether to return only unique samples. Defaults to True.
        prior (Optional[GenerativeModel], optional): optional prior if needed to generate samples. Defaults to None.
        verbose (bool): whether to give verbose updates on the progress, or print no updates if set to False. Defaults to True.
    """

    attempts = 0
    generated_samples: np.ndarray = np.zeros(0)

    # TODO: make sure samples are unique
    while len(generated_samples) < n_samples or attempts < max_attempts:
        if prior is not None:
            prior_samples = torch.tensor(prior.generate(n_samples_per_attempt)).float()
            samples = convert_to_numpy(model.generate(prior_samples))  # type: ignore
        else:
            samples = convert_to_numpy(model.generate(n_samples_per_attempt))

        if len(generated_samples) == 0:
            generated_samples = samples
        else:
            generated_samples = np.concatenate((generated_samples, samples))

        if unique:
            generated_samples = np.unique(generated_samples, axis=0)

        if verbose:
            print(
                f"Attempt {attempts + 1} / {max_attempts}. Generated {len(generated_samples)} samples."
            )
        attempts += 1

    return generated_samples[:n_samples]


def sanitize_smiles(smi):
    """Return a canonical smile representation of smi
    Parameters:
    smi (string) : smile string to be canonicalized
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful
    """
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)


def compund_to_csv(results, file_path):
    new_dataset = []
    liplinki_report = []
    qed_report = []
    sa_report = []
    logp_report = []

    for c in list(results.valid_compounds):

        try:
            mol, smi_canon, _ = sanitize_smiles(c)

            qed_report.append(compute_qed(smi_canon))
            new_dataset.append(smi_canon)
            # liplinki_report.append(compute_lipinski(c))
            logp_report.append(compute_logp(smi_canon))
            s = sascorer.calculateScore(mol)
            sa_report.append(s)
        except:
            print(c)

    data_dic = {
        "smiles": new_dataset,
        "logP": logp_report,
        "SAscore": sa_report,
        "QED": qed_report,
        # "lipinski": liplinki_report,
    }

    df = pd.DataFrame(data_dic)

    df.to_csv(file_path)


def valid_compund_to_csv(results, file_path):
    new_dataset = []
    qed_report = []
    sa_report = []
    logp_report = []

    for c in list(results):

        try:
            mol, smi_canon, _ = sanitize_smiles(c)

            qed_report.append(compute_qed(smi_canon))
            new_dataset.append(smi_canon)
            logp_report.append(compute_logp(smi_canon))
            s = sascorer.calculateScore(mol)
            sa_report.append(s)
        except:
            print(c)

    data_dic = {
        "smiles": new_dataset,
        "logP": logp_report,
        "SAscore": sa_report,
        "QED": qed_report,
        # "lipinski": liplinki_report,
    }

    df = pd.DataFrame(data_dic)

    df.to_csv(file_path)
