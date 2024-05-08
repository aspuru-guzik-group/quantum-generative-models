import inspect
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
import logging

import rdkit.Chem as rdc
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.rdmolops as rdcmo
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors

# from filter_ import maximum_ring_size, filter_phosphorus, substructure_violations

from rdkit.Chem import MolFromSmiles as smi2mol
import rdkit.Chem.rdmolops as rdcmo
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import pandas as pd
import rdkit.Chem as rdc

import os
import sys
import rdkit
from argparse import ArgumentParser
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


try:
    _mcf = pd.read_csv(os.path.join("./database/mcf.csv"))
    _pains = pd.read_csv(
        os.path.join("./database/wehi_pains.csv"), names=["smarts", "names"]
    )
    _filters = [
        Chem.MolFromSmarts(x) for x in pd.concat([_mcf,_pains],sort=True)["smarts"].values
    ]
    inf = open("./database/pains.txt", "r")
    sub_strct = [line.rstrip().split(" ") for line in inf]
    smarts = [line[0] for line in sub_strct]
    desc = [line[1] for line in sub_strct]
    dic = dict(zip(smarts, desc))

except FileNotFoundError as e:
    logging.warning(
        f"Unable to locate one or more of the filter files. Continuing without filters, this may lead to unexpected errors. Exception: {e}"
    )


def lipinski_filter(mol, max_mol_weight=800):
    try:
        # mol = Chem.MolFromSmiles(smiles)
        return (
            MolLogP(mol) <= 5
            and NumHAcceptors(mol) <= 10
            and NumHDonors(mol) <= 5
            and 300 <= ExactMolWt(mol) <= max_mol_weight
        )
    except:
        return False


def maximum_ring_size(mol):
    """
    Calculate maximum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        maximum_ring_size = 0
    else:
        maximum_ring_size = max([len(ci) for ci in cycles])
    return maximum_ring_size


def substructure_violations(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """
    violation = False
    forbidden_fragments = [
        "[S&X3]",
        "[S&X4]",
        "[S&X6]",
        "[S&X2]",
        "[S&X1]",
        "*1=**=*1",
        "*1*=*=*1",
        "*1~*=*1",
        "[F,Cl,Br]C=[O,S,N]",
        "[Br]-C-C=[O,S,N]",
        "[N,n,S,s,O,o]C[F,Cl,Br]",
        "[I]",
        "[S&X3]",
        "[S&X5]",
        "[S&X6]",
        "[B,N,n,O,S]~[F,Cl,Br,I]",
        "*=*=*=*",
        "*=[NH]",
        "[P,p]~[F,Cl,Br]",
        "SS",
        "C#C",
        "C=C=C",
        "*=*=*",
        "NNN",
        "[R3R]",
        "[R4R]",
    ]

    for ni in range(len(forbidden_fragments)):

        if mol.HasSubstructMatch(rdc.MolFromSmarts(forbidden_fragments[ni])) == True:
            violation = True
            # print('Substruct violation is: ', forbidden_fragments[ni])
            break
        else:
            continue

    return violation


def filter_phosphorus(mol):
    """
    Check for presence of phopshorus fragment
    Return True: contains proper phosphorus
    Return False: contains improper phosphorus
    """
    violation = False

    if mol.HasSubstructMatch(rdc.MolFromSmarts("[P,p]")) == True:
        if mol.HasSubstructMatch(rdc.MolFromSmarts("*~[P,p](=O)~*")) == False:
            violation = True

    return violation


def legacy_apply_filters(smi, max_mol_weight=800):
    try:
        if (
            "C-" in smi
            or "N+" in smi
            or "C+" in smi
            or "S+" in smi
            or "S-" in smi
            or "O+" in smi
        ):
            return False
        mol = smi2mol(smi)
        if mol == None:
            return False
        # Added after GDB-13 was filtered to get rid charged molecules
        if rdcmo.GetFormalCharge(mol) != 0:
            # print('Formal charge failed! Value: ', rdcmo.GetFormalCharge(mol))
            return False
        # Added after GDB-13 was filtered to get rid radicals
        elif rdcd.NumRadicalElectrons(mol) != 0:
            # print('rdcd.NumRadicalElectrons(mol) failed! Value: ', rdcd.NumRadicalElectrons(mol))
            return False
        # Filter by bridgehead atoms
        elif rdcmd.CalcNumBridgeheadAtoms(mol) > 2:
            return False
        # Filter by ring size
        elif maximum_ring_size(mol) > 8:
            return False
        # Filter by proper phosphorus
        elif filter_phosphorus(mol):
            return False
        elif substructure_violations(mol):
            return False
        elif lipinski_filter(mol, max_mol_weight) == False:
            return False
        elif rdcmd.CalcNumRotatableBonds(mol) >= 10:
            return False
        else:
            return True
    except FileNotFoundError as e:
        logging.warning(f"unable to filter in apply_filter function: {e}")


def apply_filters(smi, max_mol_weight=800):
    try:
        if (
            "C-" in smi
            or "N+" in smi
            or "C+" in smi
            or "S+" in smi
            or "S-" in smi
            or "O+" in smi
        ):
            return False
        if ("N" not in smi or "n" not in smi) and ("O" not in smi or "o" not in smi):
            return False
        mol = smi2mol(smi)
        if mol == None:
            return False
        # Added after GDB-13 was filtered to get rid charged molecules
        if rdcmo.GetFormalCharge(mol) != 0:
            # print('Formal charge failed! Value: ', rdcmo.GetFormalCharge(mol))
            return False
        # Added after GDB-13 was filtered to get rid radicals
        elif rdcd.NumRadicalElectrons(mol) != 0:
            # print('rdcd.NumRadicalElectrons(mol) failed! Value: ', rdcd.NumRadicalElectrons(mol))
            return False
        # Filter by bridgehead atoms
        elif rdcmd.CalcNumBridgeheadAtoms(mol) > 2:
            return False
        # Filter by ring size
        elif maximum_ring_size(mol) > 6:
            # TODO: check this part with Akshat
            # elif maximum_ring_size(mol) > 8: #
            return False
        # Filter by proper phosphorus
        elif filter_phosphorus(mol):
            return False
        elif substructure_violations(mol):
            return False
        elif lipinski_filter(mol, max_mol_weight) == False:
            return False
        elif len(FindAromaticRings(mol)) < 1:  # Number of aromatic rings in molecule
            return False
        elif rdcmd.CalcNumRotatableBonds(mol) >= 10:
            return False
        else:
            return True
    except:
        print("error")
        return False


def pains_filt(mol):

    for k, v in dic.items():
        subs = Chem.MolFromSmarts(k)
        if subs != None:
            if mol.HasSubstructMatch(subs):
                mol.SetProp(v, k)
    return [prop for prop in mol.GetPropNames()]


def passes_wehi_mcf(smi):
    mol = Chem.MolFromSmiles(smi)
    h_mol = Chem.AddHs(mol)
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    else:
        return True


def get_diversity(smiles_ls):

    pred_mols = [Chem.MolFromSmiles(s) for s in smiles_ls]
    pred_mols = [x for x in pred_mols if x is not None]
    pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]

    similarity = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        similarity += sum(sims)

    n = len(pred_fps)
    n_pairs = n * (n - 1) / 2
    diversity = 1 - (similarity / n_pairs)

    return diversity * 100


def IsRingAromatic(ring, aromaticBonds):
    for bidx in ring:
        if not aromaticBonds[bidx]:
            return False
    return True


def HasRingAromatic(ring, aromaticBonds):
    for bidx in ring:
        if aromaticBonds[bidx]:
            return True
    return False


def GetFusedRings(rings):
    res = rings[:]

    pool = []
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            ovl = rings[i] & rings[j]
            if ovl:
                fused = rings[i] | rings[j]
                fused.difference_update(ovl)
                pool.append(fused)
    while pool:
        res.extend(pool)
        nextRound = []
        for ringi in rings:
            li = len(ringi)
            for poolj in pool:
                ovl = ringi & poolj
                if ovl:
                    lj = len(poolj)
                    fused = ringi | poolj
                    fused.difference_update(ovl)
                    lf = len(fused)
                    if (
                        lf > li
                        and lf > lj
                        and fused not in nextRound
                        and fused not in res
                    ):
                        nextRound.append(fused)
        pool = nextRound
    return res


def FindAromaticRings(mol):
    # flag whether or not bonds are aromatic:
    aromaticBonds = [0] * mol.GetNumBonds()
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            aromaticBonds[bond.GetIdx()] = 1

    # get the list of all rings:
    ri = mol.GetRingInfo()
    # collect the ones that have at least one aromatic bond:
    rings = [set(x) for x in ri.BondRings() if HasRingAromatic(x, aromaticBonds)]

    # generate all fused ring systems from that set
    fusedRings = GetFusedRings(rings)

    aromaticRings = [x for x in fusedRings if IsRingAromatic(x, aromaticBonds)]
    return aromaticRings
