#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:06:14 2023

@author: akshat
"""

import os 
import rdkit 

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


with open('output_unique_2023_02_02.csv', 'r') as f: 
    lines = f.readlines()
lines = lines[1: ]


valid_molecules = []

for i,item in enumerate(lines):
    
    if i%100000 == 0: 
        print(i)
    
    if 'c-' in item or 'c+' in item or 'n-' in item or 'n+' in item or 's-' in item or 's+' in item: 
        continue 
    
    
    mol = Chem.MolFromSmiles(item)
    if mol == None: 
        continue
    weight = ExactMolWt(mol)
    
    if weight <= 300: 
        continue 
    
    valid_molecules.append(item)
    
with open('output_unique_2023_CORRECTED.csv', 'w') as f: 
    f.writelines(valid_molecules)