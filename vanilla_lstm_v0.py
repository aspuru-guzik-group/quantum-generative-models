from inspect import signature
import os, json, time
from datetime import datetime
from pathlib import Path
from functools import partial
from argparse import ArgumentParser
from typing import Literal, NamedTuple
import wandb
import torch
from torch import nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
import optuna
from optuna.trial import TrialState
import cloudpickle
from orquestra.qml.api import TrainCache
from orquestra.qml.models.rbm.th import RBM, TrainingParameters as RBMParams
from orquestra.qml.models.samplers.th import RandomChoiceSampler
from orquestra.qml.data_loaders import new_data_loader
from orquestra.qml.trainers import SimpleTrainer


from utils import (
    SmilesEncoding,
    SelfiesEncoding,
    generate_bulk_samples,
    truncate_smiles,
    LegacyExperiment,
    lipinski_filter,
    lipinski_hard_filter,
    compute_compound_stats,
)
from utils.lipinski_utils import (
    compute_qed,
    compute_lipinski,
    compute_logp,
    draw_compounds,
)
from models.recurrent import NoisyLSTMv3
from utils.docking import compute_array_value
from utils.data import compund_to_csv
from utils.filter import (
    apply_filters,
    filter_phosphorus,
    substructure_violations,
    maximum_ring_size,
    # lipinski_filter,
    get_diversity,
    passes_wehi_mcf,
    pains_filt,
    legacy_apply_filters,
)


import torch
from orquestra.qml.api import Batch
from models.recurrent.lstm import MolLSTM

# LOAD IBMQ account
from models.priors.loss import ExactNLLTorch




# device = "ibm_lagos"
# # device = "ibm_canberra"

# SYBA
from syba.syba import SybaClassifier
from utils.api import RewardAPI

syba = SybaClassifier()
syba.fitDefaultScore()

RDLogger.DisableLog("rdApp.*")

diversity_fn = get_diversity

BATCHSIZE_GENERETATION = 100000
ACTIVE_FILTER = False
DISABLE_PROGRESS_BAR_PRIOR = False

class TrainingArgs(NamedTuple):
    lstm_n_epochs: int
    prior_n_epochs: int
    n_compound_generation: int
    n_generation_steps: int
    prior_model: Literal[
        "QCBM", "mQCBM", "mrQCBM", "RBM", "classical", "ibm_hub_simulator"
    ]
    filter_constraint: Literal["soft", "hard"]
    n_lstm_layers: int
    embedding_dim: int
    hidden_dim: int
    prior_size: int
    n_qcbm_layers: int
    data_set_id: int
    device: Literal["cpu", "cuda", "auto"] = "auto"
    gpu_count: int = 1
    n_test_samples: int = 20_000
    batch_size: int = 128
    dataset_frac: float = 1.0
    n_samples_chemistry42: int = 30
    n_test_samples_chemistry42: int = 300
    backend_name: str = "ibmq_qasm_simulator"
    n_shots: int = 20000
    optimizer_name: str = "COBYLA"
    train_backend:str ="sw",
    sampeling_backend:str="sw"
    do_greedy_sampling: bool = False
    temprature:float = 0.5
    experiment_root:str = "/project/mghazi/experiment_results"
    random_seed:int = 42

    @classmethod
    def from_file(cls, path: str) -> "TrainingArgs":
        assert os.path.exists(path), f"File {path} does not exist"
        assert path.endswith(".json"), f"File {path} is not a json file"
        with open(path, "r") as f:
            args = json.load(f)
        return cls(**args)

    @classmethod
    def from_namespace(cls, namespace) -> "TrainingArgs":
        namespace_dict = vars(namespace)
        namespace_dict.pop("config_file", None)

        return cls(**namespace_dict)


def wma(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Returns Weighted Moving Average.

    Args:
        arr (np.ndarray): data array.
        window_size (int): window_size for computing average.
    """
    weights = np.arange(window_size)
    return np.convolve(arr, weights, "valid") / np.sum(weights)


def compute_stats_encode(data):
    new_data = []
    for data_ in data:
        try:
            reward_1 = compute_array_value(data_)
            lip = compute_lipinski(data_, mol_weight_ref=600)
            reward = np.append(lip[2], reward_1)
            new_data.append(reward)

        except:
            reward = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            new_data.append(reward)

    return torch.Tensor(new_data).sum(dim=1)


def filter_legacy(smiles_compound, max_mol_weight: float = 800):
    pass_all = []
    for smile_ in smiles_compound:
        try:
            if apply_filters(smile_, max_mol_weight=max_mol_weight):
                pass_all.append(smiles_compound)
        except:
            pass
    return pass_all


def reward_fc_legacy(data):
    new_data = []
    for data_ in data:
        try:
            if validity_fn(data_):
                reward_1 = compute_array_value(data_)
                lip = compute_lipinski(data_, mol_weight_ref=600)
                reward = np.append(lip[2], reward_1)
                new_data.append(reward)
            else:
                reward = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                new_data.append(reward)
        except:
            reward = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            new_data.append(reward)

    return torch.Tensor(new_data).sum(dim=1)


def combine_filter(
    smiles_compound, max_mol_weight: float = 800, filter_fc=apply_filters
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


def reward_fc(smiles_ls, max_mol_weight: float = 800, filter_fc=legacy_apply_filters):
    rewards = []
    for smiles_compound in smiles_ls:
        #: TODO: add wieghts for filter
        try:
            reward = 1
            if filter_fc(smiles_compound, max_mol_weight=max_mol_weight):
                reward += 15
                if passes_wehi_mcf(smiles_compound):
                    reward += 5
                    if len(pains_filt(Chem.MolFromSmiles(smiles_compound))) == 0:
                        reward += 5
                        if syba.predict(smiles_compound) > 0:
                            reward += 30

            rewards.append(reward)
        except:
            rewards.append(0)

    return torch.Tensor(rewards)


def rew_chemistry(
    smiles_list: list, api: RewardAPI, custom_w_name: str = "training_loop"
):
    workflow_ids = []
    not_submitted = []
    submitted = {}
    rewards = []
    smiles_dict = {}
    step_size = 10
    if len(smiles_list) > 10:
        for i in range(0, len(smiles_list), 10):
            smiles_ls = [smiles_["smiles"] for smiles_ in smiles_list[i : i + 10]]
            try:
                workflow_uuid = api.post_smiles(
                    name=f"{custom_w_name}_{i}_{i+10}",
                    mpo_score_definition_id=0,
                    smiles_list=smiles_ls,
                )
                submitted[workflow_uuid] = smiles_ls
                print(i, i + 10)
                submited_flag = True
            except:
                not_submitted.append(smiles_ls)
                rewards.append(step_size * [-1.6])
                submited_flag = False

            if submited_flag:
                try:
                    status = api.get_workflow_status(workflow_uuid)
                    while status != "success":
                        time.sleep(10)
                        status = api.get_workflow_status(workflow_uuid)

                    results = api.get_workflow_results(workflow_uuid)
                    for reward_, key_ in zip(results, list(range(i, i + 10))):
                        if reward_["filters_passed"]:
                            rewards.append(4 * (reward_["main_reward"] + 1))

                        else:
                            rewards.append(-1.6)
                        smiles_dict[key_] = {
                            "filters_passed": reward_["filters_passed"],
                            "ROMol_was_valid": reward_["ROMol_was_valid"],
                            "smiles": reward_["smiles"],
                            "reward": reward_["main_reward"],
                        }
                except:
                    print(f"{workflow_uuid} pulling results is faled!")
                    rewards.append(step_size * [-1.6])
    else:
        smiles_ls = [smiles_["smiles"] for smiles_ in smiles_list]
        api.post_smiles(
            name="training_loop",
            mpo_score_definition_id=0,
            smiles_list=smiles_ls,
        )
        try:
            status = api.get_workflow_status(workflow_uuid)
            while status != "success":
                time.sleep(5)
                status = api.get_workflow_status(workflow_uuid)

            results = api.get_workflow_results(workflow_uuid)
            for reward_, key_ in zip(results, list(range(i, len(results)))):
                if reward_["filters_passed"]:
                    rewards.append(4 * (reward_["main_reward"] + 1))
                else:
                    rewards.append(-1.6)
                smiles_dict[key_] = {
                    "filters_passed": reward_["filters_passed"],
                    "ROMol_was_valid": reward_["ROMol_was_valid"],
                    "smiles": reward_["smiles"],
                    "reward": reward_["main_reward"],
                }
        except:
            print(f"{workflow_uuid} pulling results is faled!")
            rewards.append(step_size * [-1.6])
    print(rewards)
    return smiles_dict, rewards


# save in file:
def save_obj(obj, file_path):
    with open(file_path, "wb") as f:
        r = cloudpickle.dump(obj, f)
    return r


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj


# inputs
argparser = ArgumentParser()

argparser.add_argument(
    "--config_file",
    type=str,
    help="Path to config file for training. Manual configuration will take priority over values in config file.",
    default="setting_vanila_lstm_v0.json",
)
argparser.add_argument(
    "--lstm_n_epochs", type=int, help="Number of epochs to train LSTM for"
)
argparser.add_argument(
    "--prior_n_epochs", type=int, help="Number of epochs to train prior for"
)
argparser.add_argument(
    "--n_compound_generation", type=int, help="Number of compounds to generate"
)
argparser.add_argument(
    "--n_generation_steps", type=int, help="Number of generation steps to take"
)
argparser.add_argument(
    "--prior_model",
    type=str,
    help="Name of prior model to use.",
    choices=["QCBM", "mQCBM", "mrQCBM", "RBM", "classical", "ibm_hub_simulator"],
    default=None,
)
argparser.add_argument(
    "--filter_constraint",
    type=str,
    help="Name of filter constraint to use.",
    choices=["hard", "soft"],
)
argparser.add_argument("--n_lstm_layers", type=int, help="Number of layers in LSTM")
argparser.add_argument("--embedding_dim", type=int, help="Embedding dimension")
argparser.add_argument("--hidden_dim", type=int, help="Hidden dimension")
argparser.add_argument(
    "--prior_size", type=int, help="Dimension of samples generated by prior."
)
argparser.add_argument("--n_qcbm_layers", type=int, help="Number of layers in the QCBM")
argparser.add_argument(
    "--data_set_id", type=int, help="ID of data set to use.", default=0
)
argparser.add_argument(
    "--device",
    type=str,
    help="Device to use for training.",
    choices=["cpu", "cuda", "auto"],
    default="auto",
)
argparser.add_argument(
    "--gpu_count", type=int, help="Number of GPUs to use, if available.", default=1
)
argparser.add_argument(
    "--n_test_samples",
    type=int,
    help="Number of test samples to generate",
    default=20_000,
)
argparser.add_argument(
    "--batch_size",
    type=int,
    help="Number of samples per batch (per GPU if multiple available).",
    default=128,
)
argparser.add_argument(
    "--dataset_frac",
    type=float,
    help="Fraction of full dataset to train on.",
    default=1.0,
)
argparser.add_argument(
    "--n_samples_chemistry42",
    type=int,
    help="Number of test samples to generate",
    default=50,
)
argparser.add_argument(
    "--n_test_samples_chemistry42",
    type=int,
    help="Number of test samples to generate",
    default=300,
)
argparser.add_argument(
    "--n_shots",
    type=int,
    help="Number of shots on the quantum HW/SIM",
    default=20000,
)
argparser.add_argument(
    "--backend_name",
    type=str,
    help="backend_name name",
    default="ibmq_qasm_simulator",
)
argparser.add_argument(
    "--optimizer_name",
    type=str,
    help="optimizer_name name",
    default="COBYLA",
)
argparser.add_argument(
    "--train_backend",
    type=str,
    help="train_backend name [sw,hw]",
    default="sw",
)
argparser.add_argument(
    "--sampeling_backend",
    type=str,
    help="sampeling_backend [sw,hw]",
    default="sw",
)
argparser.add_argument(
    "--do_greedy_sampling",
    type=str,
    help="do_greedy_sampling [True,False]",
    default=False,
)
argparser.add_argument(
    "--temprature",
    type=float,
    help="temprature [0.0,...,1.0]",
    default=1.0,
)

argparser.add_argument(
    "--experiment_root",
    type=str,
    help="experiment_root [/project/mghazi/experiment_results]",
    default= "/project/mghazi/experiment_results",
)
argparser.add_argument(
    "--random_seed",
    type=int,
    help="random_seed [42]",
    default= 42,
)

namespace, _ = argparser.parse_known_args()
if namespace.config_file is not None:
    args = TrainingArgs.from_file(namespace.config_file)
else:
    args = TrainingArgs.from_namespace(namespace)

# get values from args
max_mol_weight = 800
prior_hidden_layer = 10

device = args.device
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

prior_sample_size = args.prior_size
lstm_n_batch_size = args.batch_size
data_set = args.data_set_id
filter_constraint = args.filter_constraint
prior_model = args.prior_model
n_qcbm_layers = args.n_qcbm_layers
hidden_dim = args.hidden_dim
embedding_dim = args.embedding_dim
n_lstm_layers = args.n_lstm_layers
lstm_n_epochs = args.lstm_n_epochs
prior_n_epochs = args.prior_n_epochs
n_test_samples = args.n_test_samples
n_compound_generation = args.n_compound_generation
n_generation_steps = args.n_generation_steps
dataset_frac = args.dataset_frac
N_SAMPLES_CHEMISTRY42 = args.n_samples_chemistry42
N_TEST_SAMPLES_CHEMISTRY42 = args.n_test_samples_chemistry42
backend_name = args.backend_name
n_shots = args.n_shots
optimizer_name = args.optimizer_name
train_backend = args.train_backend
sampeling_backend = args.sampeling_backend
do_greedy_sampling = args.do_greedy_sampling
temprature = args.temprature
experiment_root = args.experiment_root
random_seed = args.random_seed  
print(f"\n**TRAINING CONFIGURATION**: {args}\n")

# path_to_dataset = "data/KRAS_G12D/KRAS_G12D_inhibitors_update2023.csv"
dataset_by_id = {
    0: "data/KRAS_G12D/KRAS_G12D_inhibitors_update2023.csv",
    1: "data/KRAS_G12D/initial_dataset.csv",
    2: "data/KRAS_G12D/initial_data_with_chemistry42_syba_merged_v2.csv",
    3: "data/KRAS_G12D/initial_data_set_with_100k_hits.csv",
    4: "data/merged_dataset/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv",
}
path_to_dataset = dataset_by_id.get(data_set, None)
if path_to_dataset is None:
    raise ValueError(f"Invalid data set id: {data_set}")

path_to_model_weights = None


if filter_constraint == "hard":
    filter_fc = partial(combine_filter, max_mol_weight=max_mol_weight)
    rew_fc = reward_fc
elif filter_constraint == "soft":
    filter_fc = partial(
        combine_filter,
        max_mol_weight=max_mol_weight,
        filter_fc=legacy_apply_filters,
    )
    rew_fc = partial(
        reward_fc, max_mol_weight=max_mol_weight, filter_fc=legacy_apply_filters
    )
else:
    raise ValueError("Invalid filter constraint.")

run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")

experiment = LegacyExperiment(run_id=f"vanila-lstm-{run_date_time}",root_dir= Path(experiment_root).resolve())
print(f"Experiment ID: {experiment.run_id}")

if data_set == 2:
    object_loaded = load_obj("data/initial_data.pkl")
    selfies = object_loaded[1]
elif data_set == 4:
    object_loaded = load_obj(
        "data/merged_dataset/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.pkl"
    )
    selfies = object_loaded[1]
else:
    selfies = SelfiesEncoding(path_to_dataset, dataset_identifier="insilico_KRAS")
print(f"Using file: {selfies._filepath}.")
print(f"Dataset identifier: {selfies.dataset_identifier}")

# TODO: optuna for
# hidden_dim=8,  # next: 64
# embedding_dim=256
# hidden_dim=128,  # next: 64 # best resulst is with hidden_dim 128

model = MolLSTM(
    vocab_size=selfies.num_emd,
    seq_len=selfies.max_length,
    sos_token_index=selfies.start_char_index,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    n_layers=n_lstm_layers
)

if path_to_model_weights is not None:
    print(f"Loading model weights from {path_to_model_weights}")
    model.load_weights(path_to_model_weights)

if device == "cuda":
    if torch.cuda.device_count() > 1 and args.gpu_count > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model._model = nn.DataParallel(model._model)
        lstm_n_batch_size = lstm_n_batch_size * torch.cuda.device_count()

    model.to_device(device)

training_parameters = {
    "n_epochs": lstm_n_epochs,
    "batch_size": lstm_n_batch_size,
}

n_epochs = training_parameters["n_epochs"]

# encoded_samples_th = torch.tensor(selfies.encoded_samples)
# data = encoded_samples_th.float()
# save_obj([data,selfies],"data/initial_data.pkl")
if data_set == 2:
    data = object_loaded[0]
elif data_set == 4:
    data = object_loaded[0]
    print(data.shape)
else:
    encoded_samples_th = torch.tensor(selfies.encoded_samples)
    data = encoded_samples_th.long()
    # save_obj(
    #     [data, selfies],
    #     "data/merged_dataset/1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.pkl",
    # )
    # exit()
decoder_fn = selfies.decode_fn
truncate_fn = truncate_smiles
validity_fn = filter_fc
train_compounds = selfies.train_samples


training_parameters.update(
    dict(
        n_test_samples=n_test_samples,
        decoder_fn_signature=str(signature(decoder_fn)),
        truncate_fn_signature=str(signature(truncate_fn)),
        validity_fn_signature=str(signature(validity_fn)),
    )
)

epoch_plot_dir = Path(experiment_root) / "epoch_plots" / experiment.run_id
epoch_plot_dir = epoch_plot_dir.resolve()

if epoch_plot_dir.exists() is False:
    os.makedirs(str(epoch_plot_dir))


dataloader = (
    new_data_loader(
        data=data, batch_size=training_parameters["batch_size"], drop_last=True
    )
    .shuffle(12345)
    .truncate(fraction=dataset_frac)
)



n_samples = data.shape[0]
# start training
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()


# wandb

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project=experiment.run_id,
    name=f"vanila-lstm",
    # Track hyperparameters and run metadata
    config={
        "args": args,
        "epoch_plot_dir": epoch_plot_dir
        }
    )

wandb.watch(model._model, log_freq=100)
save_obj(selfies, f"{epoch_plot_dir}/selfies.pkl")

generated_compunds = {}
live_model_loss = []
all_chem_42_computed = []
valid_smiles_ = []
all_compound = []
train_cache = TrainCache()
model.set_train_state()
for epoch in range(1, n_epochs + 1):
    with tqdm.tqdm(total=dataloader.n_batches) as pbar:
        pbar.set_description(f"Epoch {epoch} / {n_epochs}.")
        concat_prior_samples = []
        for batch_idx, batch in enumerate(dataloader):
            batch_result = model.train_on_batch(batch)
            train_cache.update_history(batch_result)
            pbar.set_postfix(dict(Loss=batch_result["loss"]))
            pbar.update()
        model.set_eval_state()

        
        encoded_compounds = model.generate(n_test_samples,random_seed)

        compound_stats = compute_compound_stats(
            encoded_compounds,
            decoder_fn,
            diversity_fn,
            validity_fn,
            train_compounds,
        )
        all_compound.append(compound_stats)
        # train rbm or ...
        # new_data_loader(data=data,probs=stats, batch_size=training_parameters['batch_size'], shuffle=True)
        pbar.set_postfix(
            dict(
                Loss=batch_result["loss"],
                NumUniqueGenerated=compound_stats.n_unique,
                NumValidGenerated=compound_stats.n_valid,
                NumUnseenGenerated=compound_stats.n_unseen,
                NumValidChemistry42=len(valid_smiles_),
                unique_fraction=compound_stats.unique_fraction,
                filter_fraction=compound_stats.filter_fraction,
                diversity_fraction=compound_stats.diversity_fraction,
            )
        )  # type: ignore

        # update train result so we have a history of the samples
        train_cache[str(epoch)] = dict(
            samples={
                "unique": list(compound_stats.unique_compounds),
                "valid": list(compound_stats.valid_compounds),
                "unseen": list(compound_stats.unseen_compounds),
                "chemistry42_valid": valid_smiles_,
                "unique_fraction": compound_stats.unique_fraction,
                "filter_fraction": compound_stats.filter_fraction,
                "diversity_fraction": compound_stats.diversity_fraction,
            }
        )
        generated_compunds[str(epoch)] = dict(
            samples={
                "all": list(compound_stats.all_compounds),
                "unique": list(compound_stats.unique_compounds),
                "valid": list(compound_stats.valid_compounds),
                "unseen": list(compound_stats.unseen_compounds),
                "chemistry42_valid": valid_smiles_,
                "unique_fraction": compound_stats.unique_fraction,
                "filter_fraction": compound_stats.filter_fraction,
                "diversity_fraction": compound_stats.diversity_fraction,
            }
        )

        # return model to train state
        model.set_train_state()

        # display randomly selected smiles
        rng = np.random.default_rng()
        log_t0_wandb = {
                "loss":np.mean(train_cache.history["loss"]),
                "unique_fraction":compound_stats.unique_fraction,
                "diversity_fraction":compound_stats.diversity_fraction,
                "filter_fraction":compound_stats.filter_fraction,
                "NumUniqueGenerated":compound_stats.n_unique,
                "NumValidGenerated":compound_stats.n_valid,
                "NumUnseenGenerated":compound_stats.n_unseen,
                "NumValidChemistry42":len(valid_smiles_)
            }
        try:
            selected_smiles = rng.choice(
                list(compound_stats.unseen_compounds), 20, replace=False
            )
            mols = [Chem.MolFromSmiles(smile_) for smile_ in selected_smiles]
            img = Draw.MolsToGridImage(mols, molsPerRow=20, returnPNG=False)
            log_t0_wandb.update({"dicovery": wandb.Image(img)})
            img.save(f"{epoch_plot_dir}/epoch_{epoch}.png")

        except Exception as e:
            print(f"Unable to draw molecules: {e}")
        live_model_loss.append(np.mean(train_cache.history["loss"]))

        try:
            plt.figure()
            figure_path = epoch_plot_dir / f"model_losses.png"
            plt.scatter(range(0, len(live_model_loss)), live_model_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(figure_path, bbox_inches="tight", format="png")
        except Exception as e:
            print(f"Unable to draw model loss fn in epoch {epoch}: {e}")
        try:
            data_ = {
                "all_chem_42_computed": all_chem_42_computed,
                "model_samples": encoded_compounds,
                "model": model,
                "compound_stats":compound_stats
            }
            file_name = f"mode_prior_{epoch}"
            save_obj(data_, f"{epoch_plot_dir}/{file_name}.pkl")
            
            wandb.log(log_t0_wandb)
        except Exception as e:
            print(f"Unable to save model and prior in epoch {epoch}: {e}")
        
        


# export analysis


unique_generated_samples = compound_stats.unique_compounds
valid_generated_samples = compound_stats.valid_compounds
unseen_generated_samples = compound_stats.unseen_compounds
unique_train_compounds = set(train_compounds)

print(f"Filter is {filter_constraint}, Prior model is: {prior_model}")
print(f"Number of UNIQUE compounds in TRAINING set is {len(unique_train_compounds)}.")
print(
    f"Number of UNIQUE compounds in GENERATED set is {len(unique_generated_samples)}."
)
print(
    f"Number of UNIQUE VALID samples in GENERATED set is {len(valid_generated_samples)}."
)
print(
    f"Number of UNIQUE VALID UNSEEN samples in GENERATED set is: {len(unseen_generated_samples)}."
)
print(
    f"Number of diversity fraction samples in GENERATED set is: {compound_stats.diversity_fraction}."
)
print(
    f"Number of filter fraction samples in GENERATED set is: {compound_stats.filter_fraction}."
)
print(
    f"Number of UNIQUE fraction samples in GENERATED set is: {compound_stats.unique_fraction}."
)
wandb.finish()