#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:21:49 2023

@author: akshat
"""
import os
from collections import Counter
from functools import partial
import numpy as np
import pandas as pd
import scipy.sparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors


_mcf = pd.read_csv(os.path.join('./DATA/mcf.csv'))
_pains = pd.read_csv(os.path.join('./DATA/wehi_pains.csv'), names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in _mcf.append(_pains, sort=True)['smarts'].values]

def passes_wehi_mcf(h_mol): 
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False 
    else: 
        return True

with open('./DATA/sim_uniq_2023_res_short.csv', 'r') as f: 
    lines = f.readlines()
header = lines[0]
lines = lines[1: ]
    
for i,item in enumerate(lines): 
    A = item.split(',')
    
    smi = A[0]
    filt_pass = A[-3]
    
    mol =  Chem.MolFromSmiles(smi)
    h_mol = Chem.AddHs(mol)

    checkmol = passes_wehi_mcf( h_mol )
    
    if checkmol == False: 
        raise Exception('T')
    