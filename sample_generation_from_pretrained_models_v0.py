from typing import Dict, Optional, List, Any, Sequence, Callable, Set
from inspect import signature
from dataclasses import dataclass
import os, json, random, string, math, sys
from datetime import datetime
from pathlib import Path
from functools import partial
import torch
from torch import nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from rdkit import Chem
from rdkit.Chem import Draw
from orquestra.qml.api import (
    TorchGenerativeModel,
    GenerativeModel,
    Callback,
    TrainCache,
    convert_to_numpy,
    GenerativeModel,
)
import pandas as pd
from orquestra.qml.optimizers.th import AdamConfig
from orquestra.qml.models.rbm.th import RBM, TrainingParameters as RBMParams
from orquestra.qml.models.samplers.th import RandomChoiceSampler
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.trainers import SimpleTrainer
from utils import (
    SmilesEncoding,
    SelfiesEncoding,
    generate_bulk_samples,
    DisplaySmilesCallback,
    truncate_smiles,
    Experiment,
    LegacyExperiment,
    lipinski_filter,
    lipinski_hard_filter,
    legacy_apply_filters,
    compute_compound_stats_new,
    RewardAPI,
)
from utils.lipinski_utils import (
    compute_qed,
    compute_lipinski,
    compute_logp,
    draw_compounds,
)
from models.recurrent import NoisyLSTMv3
from utils.docking import compute_array_value
from models.priors.qcbm import QCBMSamplingFunction_v2, QCBMSamplingFunction_v3
from utils.data import compund_to_csv

# nicer plots
seaborn.set()

# allows us to ignore warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
# QCBM imports
from orquestra.qml.models.qcbm.layer_builders import LineEntanglingLayerBuilder
from orquestra.qml.models.qcbm.ansatze.alternating_entangling_layers import (
    EntanglingLayerAnsatz,
)
from orquestra.qml.models.qcbm import WavefunctionQCBM
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer

# multi bases
from orquestra.qml.models.qcbm import MultiBasisWavefunctionQCBM
from orquestra.quantum.circuits import X, create_layer_of_gates
import time
from syba.syba import SybaClassifier
from utils.filter import (
    apply_filters,
    filter_phosphorus,
    substructure_violations,
    maximum_ring_size,
    # lipinski_filter,
    get_diversity,
    passes_wehi_mcf,
    pains_filt,
)
import dill
import cloudpickle
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi


base_url = "https://rip.chemistry42.com"
username = "m.ghazivakili"
password = "hJEV0jm5bgqX"
reward_api = RewardAPI(username=username, password=password, base_url=base_url)


print("started!")
diversity_fn = get_diversity
start_time = time.time()
syba = SybaClassifier()
syba.fitDefaultScore()


# save in file:
def save_obj(obj, file_path):
    with open(file_path, "wb") as f:
        r = cloudpickle.dump(obj, f)
    return r


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj

def combine_filter(
    smiles_compound, max_mol_weight: float = 800, filter_fc=legacy_apply_filters
):
    # syba imports take a while move them here to only import when needed

    pass_all = []
    i = 0

    with tqdm.tqdm(total=len(smiles_compound)) as pbar:
        for smile_ in smiles_compound:
            pbar.set_description(
                f"Filtered {i} / {len(smiles_compound)}. passed={len(pass_all)},frac={len(pass_all)/len(smiles_compound)}"
            )
            try:
                if (
                    filter_fc(smile_, max_mol_weight)
                    and smile_ not in pass_all
                    and (syba.predict(smile_) > 0)
                    and passes_wehi_mcf(smile_)
                    and (len(pains_filt(Chem.MolFromSmiles(smile_))) == 0)
                ):
                    pass_all.append(smile_)
            except Exception as e:
                print(
                    f"The following error occurred during the `combine_filter` step: {e}"
                )

            i += 1
            pbar.update()
    return pass_all

def reward_fc(smiles_ls):
    rewards = []
    for smiles_compound in smiles_ls:
        #: TODO: add wieghts for filter
        try:
            reward = 0
            if smiles_compound not in smiles_ls:
                reward += 5
                if apply_filters(smiles_compound):
                    reward += 5
                    if passes_wehi_mcf(smiles_compound):
                        reward += 5
                        if len(pains_filt(Chem.MolFromSmiles(smiles_compound))) == 0:
                            reward += 5
                            if syba.predict(smiles_compound) > 0:
                                reward += 10

            rewards.append(reward)
        except:
            rewards.append(0)

    return torch.Tensor(rewards)

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


def send_to_chemistry42(model_type, model, prior, n_samples):
    not_submitted = []
    submitted = {}
    for i in range(n_samples // 10):
        g_samples = generate_bulk_samples(
            model, 10, 10, 1, prior=prior, verbose=True, unique=True,
        )
        print(g_samples.shape)
        smiles_list = decoder_fn(g_samples)
        print(smiles_list)
        try:
            workflow_uuid = reward_api.post_smiles(f"{model_type}_{i}", 0, smiles_list)
            print(workflow_uuid)
            submitted[workflow_uuid] = smiles_list
        except:
            not_submitted.append(smiles_list)
            print("not posted!")
        print(time.sleep(5))
    return (submitted, not_submitted)

def submit_smiles_to_chemistry42(model_type, to_submit):
    not_submitted = []
    submitted = {}
    for smiles_list in to_submit:
        print(smiles_list)
        try:
            workflow_uuid = reward_api.post_smiles(f"{model_type}_{i}", 0, smiles_list)
            print(workflow_uuid)
            submitted[workflow_uuid] = smiles_list
        except:
            not_submitted.append(smiles_list)
            print("not posted!")
            # print(time.sleep(60))
        print(time.sleep(5))
    return (submitted, not_submitted)

def submit_pull_smiles_to_chemistry42(model_type, to_submit):
    not_submitted = []
    submitted = {}
    all_results = []
    for smiles_list in to_submit:
        print(smiles_list)
        try:
            workflow_uuid = reward_api.post_smiles(f"{model_type}_{i}", 0, smiles_list)
            print(workflow_uuid)
            submitted[workflow_uuid] = smiles_list
            with open("data/results/workflow_uuids.txt", "a") as file:
                file.write(f"{workflow_uuid}\n")
            status = reward_api.get_workflow_status(workflow_uuid)
            while status != "success":
                time.sleep(5)
                status = reward_api.get_workflow_status(workflow_uuid)
            results = reward_api.get_workflow_results(workflow_uuid)
            parsed_results = reward_api.parse_results(results, model_type)
            all_results.extend(parsed_results)
            
        except:
            not_submitted.append(smiles_list)
            print("not posted!")
            # print(time.sleep(60))
        print(time.sleep(5))
    results_df = pd.DataFrame(all_results)
    output_csv_path = "workflow_results_new.csv"
    results_df.to_csv(output_csv_path)
    return (submitted, not_submitted,all_results)






epoch_start=30
epoch_end=30
n_samples = 1_000_000
max_shot =10_000
n_test_samples = 20_000
new_sampleling = True
n_samples_ = n_samples
server_root = "/home/mghazi/" # /u/mghazi/
path_to_store_results = "results/new_samples_insilico"


store_dir = Path(path_to_store_results) 
store_dir = store_dir.resolve()

if store_dir.exists() is False:
    os.makedirs(str(store_dir))
    
saved_experiments={
    "sim-mqcbm":f"{server_root}/workspace/insilico-drug-discovery/experiment_results/epoch_plots/lstm-mQCBM-2023_14_06T10_42_39.812072",
    "sim-qcbm": f"{server_root}/workspace/insilico-drug-discovery/experiment_results/epoch_plots/lstm-QCBM-2023_11_06T11_35_33.240899",
    "classical":f"{server_root}/workspace/insilico-drug-discovery/experiment_results/epoch_plots/lstm-classical-2023_14_06T10_42_46.769890",
    "sim-rqcbm":f"{server_root}/workspace/insilico-drug-discovery/experiment_results/epoch_plots/lstm-mrQCBM-2023_15_06T09_38_50.912235" 
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(10*"===============")
print(f"samples {n_samples}")

print("model name,filter, div., uniqueness")
full_results = {}
for key, path in saved_experiments.items():
    for epoch in range(epoch_start,epoch_end+1):
        data = {}
        results_ = {}
    # for key, path in saved_experiments.items():
        # print(key,path)
        sub_data = {}
        data[key]=load_obj(f"{path}/mode_prior_{epoch}.pkl")
        try:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model.
            selfies = data[key]["selfies"]
            model = data[key]["model"]
            prior = data[key]["prior"]
            model_samples = data[key]["model_samples"]
            prior_samples = data[key]["prior_samples"]
            prior_x = data[key]["prior_x"]
            prior_y = data[key]["prior_y"]
            all_chem_42_computed = data[key]["all_chem_42_computed"]
            compound_stats = data[key]["compound_stats"]
            sub_data = {
                "all_chem_42_computed": all_chem_42_computed,
                "prior_samples": prior_samples,
                "model_samples": model_samples,
                "selfies": selfies,
                "prior": prior,
                "model": model,
                "prior_x":prior_x,
                "prior_y":prior_y,
                "compound_stats":compound_stats
            }
            unique_train_compounds = set(data[key]["selfies"].df.smiles.values.tolist())
            model.to_device(device)

            decoder_fn = selfies.decode_fn
            validity_fn = partial(combine_filter, max_mol_weight=800)
            train_compounds = selfies.train_samples
            
            if new_sampleling:
                prior_samples = torch.tensor(prior.generate(n_test_samples)).float()
                model_samples = model.generate(prior_samples)

            smiles_ = decoder_fn(model_samples)
            all_smiles_generated=[]
            for smi_ in smiles_:
                ss = sanitize_smiles(smi_)
                if ss[2]:
                    all_smiles_generated.append(ss[1])  
            
            canonical_smiles = set(all_smiles_generated)
            u_canonical_smiles_list = list(canonical_smiles)    
            canonical_smiles_list = list(canonical_smiles)                     
            
            all_smile_filtered = combine_filter(u_canonical_smiles_list)  
            results_[epoch]={
                "filter_fraction":compound_stats.filter_fraction,
                "diversity_fraction":compound_stats.diversity_fraction,
                "unique_fraction":compound_stats.unique_fraction,
                "all_smile_filtered":all_smile_filtered,
                "all_unique_generated_samples:":canonical_smiles,
                
                
            }
            print(key,compound_stats.filter_fraction,compound_stats.diversity_fraction,compound_stats.unique_fraction)
            prior.generate(10)
            
        except Exception as e:
            print(f"Unable to open model and prior in {sys.argv[1]}: {e}")
            exit()

    full_results[key]  = results_
    

for key, path in saved_experiments.items():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selfies = data[key]["selfies"]
    model = data[key]["model"]
    model.to_device(device)
    prior = data[key]["prior"]
    model_samples = data[key]["model_samples"]
    prior_samples = data[key]["prior_samples"]
    prior_x = data[key]["prior_x"]
    prior_y = data[key]["prior_y"]
    all_chem_42_computed = data[key]["all_chem_42_computed"]
    compound_stats = data[key]["compound_stats"]

    decoder_fn = selfies.decode_fn
    validity_fn = partial(combine_filter, max_mol_weight=800)
    train_compounds = selfies.train_samples


    smiles_list=[]

    for i in range(0,n_samples_,max_shot): 
        prior_samples = torch.tensor(prior.generate(max_shot)).float()
        encoded_smiles_ = model.generate(prior_samples)

        smiles_list.extend(decoder_fn(encoded_smiles_))
        
    all_smiles_generated=[]
    for smi_ in smiles_list:
        ss = sanitize_smiles(smi_)
        if ss[2]:
            all_smiles_generated.append(ss[1])      

    canonical_smiles = set(all_smiles_generated)
    u_canonical_smiles_list = list(canonical_smiles)    
    canonical_smiles_list = list(canonical_smiles) 
    data = {
        "smiles":list(canonical_smiles_list)
    }
    df = pd.DataFrame(data)
    run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")
    file_name = saved_experiments[key].split("/")[-2]
    df.to_csv(f"{store_dir}/{key}_{file_name}_{epoch}_{run_date_time}.csv")
    # filter smiles   
    all_smile_filtered = combine_filter(u_canonical_smiles_list)
    # stats = compute_compound_stats_new(smiles_list,diversity_fn,validity_fn,unique_train_compounds)
    data = {
        "smiles":list(all_smile_filtered)
    }

    df = pd.DataFrame(data)
    run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")
    file_name = saved_experiments[key].split("/")[-2]
    df.to_csv(f"{store_dir}/{key}_{file_name}_{epoch}_{run_date_time}_f.csv")
