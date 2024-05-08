import os 
import time 
import multiprocessing
import inspect
import tempfile
from pathlib import Path

from rdkit import Chem 
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
# from filter_ import maximum_ring_size, filter_phosphorus, substructure_violations

from rdkit.Chem import MolFromSmiles as smi2mol
import rdkit.Chem.rdmolops as rdcmo
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd

import rdkit.Chem as rdc
from syba.syba import SybaClassifier


def lipinski_filter(mol):
    try: 
        return MolLogP(mol) <= 5 and NumHAcceptors(mol) <= 10 and NumHDonors(mol) <= 5 and 300 <= ExactMolWt(mol) <= 800
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


def IsRingAromatic(ring,aromaticBonds):
    for bidx in ring:
        if not aromaticBonds[bidx]:
            return False
    return True

def HasRingAromatic(ring,aromaticBonds):
    for bidx in ring:
        if aromaticBonds[bidx]:
            return True
    return False

def GetFusedRings(rings):
    res=rings[:]
    
    pool=[]
    for i in range(len(rings)):
        for j in range(i+1,len(rings)):
            ovl=rings[i]&rings[j]
            if ovl:
                fused=rings[i]|rings[j]
                fused.difference_update(ovl)
                pool.append(fused)
    while pool:
        res.extend(pool)
        nextRound=[]
        for ringi in rings:
            li=len(ringi)
            for poolj in pool:
                ovl = ringi&poolj
                if ovl:
                    lj = len(poolj)
                    fused = ringi|poolj
                    fused.difference_update(ovl)
                    lf = len(fused)
                    if lf>li and lf>lj and fused not in nextRound and fused not in res:
                        nextRound.append(fused)
        pool = nextRound
    return res


def FindAromaticRings(mol):
    # flag whether or not bonds are aromatic:
    aromaticBonds = [0]*mol.GetNumBonds()
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            aromaticBonds[bond.GetIdx()]=1

    # get the list of all rings:
    ri = mol.GetRingInfo()
    # collect the ones that have at least one aromatic bond:
    rings=[set(x) for x in ri.BondRings() if HasRingAromatic(x,aromaticBonds)]

    # generate all fused ring systems from that set
    fusedRings=GetFusedRings(rings)

    aromaticRings = [x for x in fusedRings if IsRingAromatic(x,aromaticBonds)]
    return aromaticRings

# m = Chem.MolFromSmiles('C1=CC2=C(C=C1)C=C1C=C3C=C4C=C5C(C=CC=C5C5=CC=CC6=C5C=CC=C6)=CC4=CC3=CC1=C2')
# A = FindAromaticRings(m)
# raise Exception('T')

def substructure_violations(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """
    violation = False
    forbidden_fragments = [
        '[S&X3]', 
        '[S&X4]',
        '[S&X6]',
        '[S&X2]',
        '[S&X1]',
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
    ]

    for ni in range(len(forbidden_fragments)):

        if mol.HasSubstructMatch(rdc.MolFromSmarts(forbidden_fragments[ni])) == True:
            violation = True
            # print('Substruct violation is: ', forbidden_fragments[ni])
            break
        else:
            continue

    return violation

# m = Chem.MolFromSmiles('SC1=CC=CC2=CC3=CC4=CC5=CC6=C(C=CC=C6)C=C5C=C4C=C3C=C12')
# A = substructure_violations(m)

# A = len(Chem.FindPotentialStereo(m))

# raise Exception('T')

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
    

def apply_filters(smi):

    try: 
        if 'C-' in smi or 'N+' in smi or 'C+' in smi or 'S+' in smi or 'S-' in smi or 'O+' in smi: 
            return False
        if ('N' not in smi or 'n' not in smi) and ('O' not in smi or 'o' not in smi): 
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
            return False
        # Filter by proper phosphorus
        elif filter_phosphorus(mol):
            return False
        elif substructure_violations(mol):
            return False
        elif lipinski_filter(mol) == False:
            return False
        elif len(FindAromaticRings(mol)) < 1: # Number of aromatic rings in molecule
            return False
        elif rdcmd.CalcNumRotatableBonds(mol) >= 10: 
            return False
        # elif len(len(Chem.FindPotentialStereo(mol))) > 10: 
        #     return False 
        else: 
            return True 
    except: 
        return False 
    
    
# with open('KRAS_G12D_inhibitors_update2023.csv', 'r') as f: 
#     smiles_all = f.readlines()
# smiles_all = smiles_all[1: ]



    