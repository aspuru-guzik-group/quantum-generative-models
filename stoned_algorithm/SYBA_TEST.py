#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:37:01 2023

@author: akshat
"""
import time
from rdkit import Chem
from syba.syba import SybaClassifier

start_time = time.time()
syba = SybaClassifier()
syba.fitDefaultScore()
print('Syba fitting time: ', time.time()-start_time)


with open('./DATA/sim_uniq_2023_res_short.csv', 'r') as f: 
    lines = f.readlines()
header = lines[0]
lines = lines[1: ]

lines = lines[0: 100000] # TODO: Random subset!
    

syba_pass_chemistry42_fail = 0
syba_pass_chemistry42_pass = 0
total_chemistry42_pass = 0

for i,item in enumerate(lines): 
    
    if i%10000 == 0: 
        print('On: {}/{}'.format(i, len(lines)))
    
    A = item.split(',')
    
    smi = A[0]
    filt_pass = A[-3]
    
    if filt_pass == 'True': 
        total_chemistry42_pass += 1
    
    mol =  Chem.MolFromSmiles(smi)
    syba_score = syba.predict(mol=mol)
    
    if syba_score > 0 and filt_pass == 'False': 
        syba_pass_chemistry42_fail += 1
    if syba_score > 0 and filt_pass == 'True': 
        syba_pass_chemistry42_pass += 1
        








