from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole


inf = open("./DATA/pains.txt", "r")
sub_strct = [ line.rstrip().split(" ") for line in inf ]
smarts = [ line[0] for line in sub_strct]
desc = [ line[1] for line in sub_strct]
dic = dict(zip(smarts, desc))

def pains_filt(mol):
    """
    >>> mol = Chem.MolFromSmiles("c1ccccc1N=Nc1ccccc1")
    >>> checkmol = pains_filt( mol )
    >>> props = [ prop for prop in checkmol.GetPropNames() ]
    >>> props[0]
    'azo_A(324)'
    """

    for k,v in dic.items():
        subs = Chem.MolFromSmarts( k )
        if subs != None:
            if mol.HasSubstructMatch( subs ):
                mol.SetProp(v,k)
    return mol



with open('./DATA/sim_uniq_2023_res_short.csv', 'r') as f: 
    lines = f.readlines()
header = lines[0]
lines = lines[1: ]
    
for i,item in enumerate(lines): 
    A = item.split(',')
    
    smi = A[0]
    filt_pass = A[-3]
    
    mol =  Chem.MolFromSmiles(smi)
    checkmol = pains_filt( mol )
    props = [ prop for prop in checkmol.GetPropNames() ]

    if len(props) > 0: 
        raise Exception('T')
    
