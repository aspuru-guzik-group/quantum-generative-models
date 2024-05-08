import typing as t
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Set, Sequence
import numpy as np
import pandas as pd
import selfies as sf
import torch
from orquestra.qml.api import Tensor, convert_to_numpy
from orquestra.qml.data_loaders import SizedDataLoader, new_data_loader
from PIL import Image
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from scipy.special import softmax

from .mol_methods import *

from .docking import compute_array_value
from .lipinski_utils import compute_lipinski


class PaddingOptions(Enum):
    before = "before"
    after = "after"


class Smiles:
    def __init__(self, path_to_df: str, padding_token: str = " ") -> None:
        self.path_to_df = path_to_df
        self.df = pd.read_csv(path_to_df)
        self.smiles = np.asarray(self.df.smiles)
        self.alphabet = list(set("".join(self.smiles)))
        self.alphabet.append(padding_token)
        self.padding_token = padding_token

    @property
    def max_length(self) -> int:
        """Returns the length of the longest smile sequence."""
        return len(max(self.smiles, key=len))


class Selfies:
    def __init__(
        self,
        path_to_df: str,
        dataset_identifier: str,
        padding_symbol: str = "[nop]",
        padding_mode: PaddingOptions = PaddingOptions.after,
        lable: str = "binding_ic50",
    ) -> None:
        """Initialize a wrapper class for SELFIES.

        Args:
            selfies (t.List[str]): a list of SELFIES strings.
            padding_token (str, optional): symbol to use for padding. Defaults to "[nop]".
            padding_mode (PaddingOptions, optional): whether to add padding before or after actual sequence.
                Defaults to PaddingOptions.before.
        """
        self.dataset_identifier = dataset_identifier
        self._filepath = path_to_df
        self.df = pd.read_csv(self._filepath)
        self.selfies = self.df.selfies.values.tolist()
        if lable == "false":
            self.Y = self.Y_orginal = np.ones(len(self.selfies))
        else:
            self.Y_orginal = self.df[lable].to_numpy()
            self.Y = np.nan_to_num(self.Y_orginal, np.nanmean(self.Y_orginal))
            self.low_bond = np.percentile(self.Y, 25)
            self.high_bond = np.percentile(self.Y, 75)
            self.Y[self.Y > self.high_bond] = self.high_bond
            self.Y[self.Y < self.low_bond] = self.low_bond

        alphabet_set = sf.get_alphabet_from_selfies(self.selfies)
        alphabet_set.add(padding_symbol)
        self.alphabet = list(alphabet_set)
        self.padding_symbol = padding_symbol
        self.padding_mode = padding_mode.value
        self.num_emd = len(self.alphabet)
        self.symbol_to_idx = dict((c, i) for i, c in enumerate(self.alphabet))
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}

    @property
    def max_length(self) -> int:
        """Returns the length of the longest selfie sequence."""
        return max(sf.len_selfies(str(s)) for s in self.selfies)

    @property
    def padded_selfies(self) -> t.List[str]:
        """Returns a list of selfies padded such that
        every string has the length of the longest string.
        """
        # faster appends and pops at edges
        padded_selfies = deque()
        for selfie in self.selfies:
            padded_selfies.append(self.pad_selfie(selfie))

        return list(padded_selfies)

    @property
    def n_selfies(self) -> int:
        return len(self.selfies)

    @property
    def n_symbols(self) -> int:
        return len(self.alphabet)

    @property
    def probs(self) -> np.ndarray:
        return softmax(1 / self.Y)

    def get_symbol_at_index(self, idx: int) -> str:
        """Returns symbol mapped to specified index."""
        return self.idx_to_symbol[idx]

    def get_index_of_symbol(self, symbol: str) -> int:
        return self.symbol_to_idx[symbol]

    def split_selfie(self, selfie: str) -> t.List[str]:
        """Split selfie string into its constituent symbols"""
        return list(sf.split_selfies(selfie))

    def pad_selfie(self, selfie: str) -> str:
        """Add padding to a selfie such that the length of the padded selfie,
        matches that of the longest selfie in the dataset.
        """
        n_padding_tokens = self.max_length - sf.len_selfies(selfie)
        padding = self.padding_symbol * n_padding_tokens

        if self.padding_mode == "before":
            padded_selfie = padding + selfie
        elif self.padding_mode == "after":
            padded_selfie = selfie + padding
        else:
            raise ValueError(f"Invalid padding mode {self.padding_mode}.")

        return padded_selfie

    def one_hot_encoded(self) -> list:
        """One-hot encode selfies and return in the form of a Numpy Array."""
        one_selfies_array = (
            []
        )  # np.zeros((self.n_selfies, self.max_length, self.n_symbols))
        for selfie_idx, selfie in enumerate(self.padded_selfies):
            encoded_selfie = sf.batch_selfies_to_flat_hot(
                self.split_selfie(selfie),
                self.symbol_to_idx,
            )
            # encoded_selfie = sf.selfies_to_encoding(selfie,self.symbol_to_idx)
            one_selfies_array.append(np.asarray([encoded_selfie]))

        return one_selfies_array

    def dense_encoded(self) -> np.ndarray:
        """Returns an array of densely encoded Selfies. In other words
        every selfie is converted to a sequence of integers, where each integer maps
        back to a symbol in the alphabet.
        """
        dense_selfies_array = np.zeros((self.n_selfies, self.max_length))
        for selfie_idx, selfie in enumerate(self.padded_selfies):
            tokenized_selfie = self.split_selfie(selfie)
            for pos_idx, symbol in enumerate(tokenized_selfie):
                dense_selfies_array[selfie_idx, pos_idx] = self.get_index_of_symbol(
                    symbol
                )

        return dense_selfies_array


class SelfiesEncoding:
    def __init__(
        self,
        filepath: str,
        dataset_identifier: str,
        start_char: str = "[^]",
        pad_char: str = "[nop]",
        max_length: t.Optional[int] = None,
    ):
        """

        Args:
            filepath (str): path to file with smiles data.
            max_length (t.Optional[int], optional): an optional argument to specify the maximum length of sequences.
                If not specified, the length of the 1.5 times the longest sequence in the provided file will be used.
        """
        self.dataset_identifier = dataset_identifier
        self._filepath = filepath
        self.df = pd.read_csv(self._filepath)
        train_samples = []
        for com in self.df.smiles.tolist():
            if "." in com:
                len_0 = len(com.split(".")[0])
                len_1 = len(com.split(".")[1])
                if len_0 > len_1:
                    com = com.split(".")[0]
                elif len_0 <= len_1:
                    com = com.split(".")[1]
            encoded_smiles = sf.encoder(com)
            if encoded_smiles is not None:
                train_samples.append(encoded_smiles)
        self.train_samples = train_samples
        # print(len(self.train_samples))
        alphabet_set = sf.get_alphabet_from_selfies(self.train_samples)
        alphabet_set.add(pad_char)
        alphabet_set.add(start_char)
        self.alphabet = list(alphabet_set)
        # mapping char -> index and mapping index -> char
        self.char_to_index = dict((c, i) for i, c in enumerate(self.alphabet))
        self.index_to_char = {v: k for k, v in self.char_to_index.items()}

        self.num_emd = len(self.char_to_index)
        self._pad_char = pad_char
        self._start_char = start_char
        self.data_length = max(map(len, self.train_samples))

        fallback_max_length = int(len(max(self.train_samples, key=len)) * 1.5)
        self._max_length = max_length if max_length is not None else fallback_max_length
        self.track_strings = []

        # self.encoded_samples_size = len(self.encoded_samples)

    def get_char_at(self, index: int) -> str:
        return self.index_to_char[index]

    def get_chars_at(self, indices: t.List[int]) -> t.List[str]:
        return [self.get_char_at(index) for index in indices]

    def get_index_of(self, char: str) -> int:
        return self.char_to_index[char]

    def pad_selfie(self, selfie: str) -> str:
        """Add padding to a selfie such that the length of the padded selfie,
        matches that of the longest selfie in the dataset.
        """
        n_padding_tokens = self.max_length - sf.len_selfies(selfie)
        padding = self.pad_char * n_padding_tokens
        padded_selfie = selfie + padding

        return padded_selfie

    @property
    def padded_selfies(self) -> t.List[str]:
        """Returns a list of selfies padded such that
        every string has the length of the longest string.
        """
        # faster appends and pops at edges
        padded_selfies = deque()
        for selfie in self.train_samples:
            padded_selfies.append(self.pad_selfie(selfie))

        return list(padded_selfies)

    @property
    def pad_char(self) -> str:
        return self._pad_char

    @property
    def start_char(self) -> str:
        return self._start_char

    @property
    def pad_char_index(self) -> int:
        return self.get_index_of(self.pad_char)

    @property
    def start_char_index(self) -> int:
        return self.get_index_of(self.start_char)

    @property
    def max_length(self) -> int:
        return self._max_length

    # @property
    # def encoded_samples(self) -> np.ndarray:
    #     # Encode samples
    #     to_use = [
    #         sample
    #         for sample in self.train_samples
    #         if mm.verified_and_below(sample, self.max_length)
    #     ]
    #     encoded_samples = [
    #         mm.encode(sam, self.max_length, self.char_to_index) for sam in to_use
    #     ]
    #     return np.asarray(encoded_samples)
    @property
    def encoded_samples(self) -> np.ndarray:
        # Encode samples
        # to_use = [
        #     sample
        #     for sample in self.train_samples
        #     if mm.verified_and_below(sample, self.max_length)
        # ]
        encoded_samples = [
            sf.selfies_to_encoding(sel, self.char_to_index, enc_type="label")
            for sel in self.padded_selfies
        ]
        return np.asarray(encoded_samples)

    # @property
    # def one_hot_encoded_samples(self) -> np.ndarray:
    #     encoded_samples = self.encoded_samples
    #     n_samples = encoded_samples.shape[0]
    #     one_hot_encoding = np.zeros((n_samples, self.max_length, self.num_emd))
    #     for i_seq, seq in enumerate(encoded_samples):
    #         for i_element, element in enumerate(seq):
    #             one_hot_encoding[i_seq, i_element, element] = 1.0

    #     return one_hot_encoding

    # def decode_one_hot_smiles(self, smiles_one_hot: Tensor) -> t.List[str]:
    #     encoded_smiles_list = convert_to_numpy(smiles_one_hot).argmax(axis=2)
    #     return self.decode_smiles(encoded_smiles_list)

    def digit_to_selfies(self, encoded_selfies):
        selfies = sf.encoding_to_selfies(
            encoded_selfies, self.index_to_char, enc_type="label"
        )
        return selfies

    def decode_fn(self, encoded_selfies: Tensor) -> t.List[str]:
        # smiles are going to be one-hot encoded
        encoded_sf_list = convert_to_numpy(encoded_selfies).tolist()
        self.track_strings = []
        decoded_sf_list = list()
        for encoded_sf in encoded_sf_list:
            decoded_sf = self.digit_to_selfies(encoded_sf)
            if self._start_char in decoded_sf:
                decoded_sf = decoded_sf.replace(self._start_char, "")
            decoded_smile = sf.decoder(decoded_sf)
            decoded_sf_list.append(decoded_smile)
        return decoded_sf_list

    def decode_char_selfies(self, encoded_selfies: Tensor) -> t.List[str]:
        # smiles are going to be one-hot encoded
        encoded_sf_list = convert_to_numpy(encoded_selfies).tolist()
        decoded_sf_list = list()
        sf.decoder()
        for encoded_sf in encoded_sf_list:
            decoded_smile = sf.decoder(encoded_sf)
            decoded_sf_list.append(decoded_smile)
        return decoded_sf_list

    def draw_smiles(self, smiles: str, molsPerRow: int = 0) -> Image:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mols_filtered = [mol for mol in mols if mol is not None]
        if len(mols_filtered) == 0:
            raise RuntimeError("No Valid smiles were provided.")

        if molsPerRow <= 0:
            molsPerRow = len(mols)

        img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, returnPNG=False)
        return img


class SmilesEncoding:
    def __init__(
        self,
        filepath: str,
        dataset_identifier: str,
        start_char: str = "^",
        pad_char: str = "_",
        max_length: t.Optional[int] = None,
    ):
        """

        Args:
            filepath (str): path to file with smiles data.
            max_length (t.Optional[int], optional): an optional argument to specify the maximum length of sequences.
                If not specified, the length of the 1.5 times the longest sequence in the provided file will be used.
        """
        self.dataset_identifier = dataset_identifier
        self._filepath = filepath
        self.train_samples = load_train_data(filepath)

        # mapping char -> index and mapping index -> char
        self.char_to_index, self.index_to_char = build_vocab(
            self.train_samples, pad_char=pad_char, start_char=start_char
        )

        self.num_emd = len(self.char_to_index)
        self._pad_char = pad_char
        self._start_char = start_char
        self.data_length = max(map(len, self.train_samples))

        fallback_max_length = int(len(max(self.train_samples, key=len)) * 1.5)
        self._max_length = max_length if max_length is not None else fallback_max_length

        self.encoded_samples_size = len(self.encoded_samples)

    def get_char_at(self, index: int) -> str:
        return self.index_to_char[index]

    def get_chars_at(self, indices: t.List[int]) -> t.List[str]:
        return [self.get_char_at(index) for index in indices]

    def get_index_of(self, char: str) -> int:
        return self.char_to_index[char]

    @property
    def pad_char(self) -> str:
        return self._pad_char

    @property
    def start_char(self) -> str:
        return self._start_char

    @property
    def pad_char_index(self) -> int:
        return self.get_index_of(self.pad_char)

    @property
    def start_char_index(self) -> int:
        return self.get_index_of(self.start_char)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def encoded_samples(self) -> np.ndarray:
        # Encode samples
        to_use = [
            sample
            for sample in self.train_samples
            if verified_and_below(sample, self.max_length)
        ]
        encoded_samples = [
            encode(sam, self.max_length, self.char_to_index) for sam in to_use
        ]
        return np.asarray(encoded_samples)

    @property
    def one_hot_encoded_samples(self) -> np.ndarray:
        encoded_samples = self.encoded_samples
        n_samples = encoded_samples.shape[0]
        one_hot_encoding = np.zeros((n_samples, self.max_length, self.num_emd))
        for i_seq, seq in enumerate(encoded_samples):
            for i_element, element in enumerate(seq):
                one_hot_encoding[i_seq, i_element, element] = 1.0

        return one_hot_encoding

    def decode_oh_smiles(self, smiles_one_hot: Tensor) -> t.List[str]:
        encoded_smiles_list = convert_to_numpy(smiles_one_hot).argmax(axis=2)
        return self.decode_smiles(encoded_smiles_list)

    def decode_smiles(self, encoded_smiles: Tensor) -> t.List[str]:
        # smiles are going to be one-hot encoded
        encoded_smiles_list = convert_to_numpy(encoded_smiles).tolist()
        decoded_smiles_list = list()
        for encoded_smile in encoded_smiles_list:
            decoded_smile = decode(encoded_smile, self.index_to_char)
            decoded_smiles_list.append(decoded_smile)
        return decoded_smiles_list

    def draw_smiles(self, smiles: str, molsPerRow: int = 0) -> Image:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mols_filtered = [mol for mol in mols if mol is not None]
        if len(mols_filtered) == 0:
            raise RuntimeError("No Valid smiles were provided.")

        if molsPerRow <= 0:
            molsPerRow = len(mols)

        img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, returnPNG=False)
        return img

    def create_data_loader(
        self,
        batch_size: int,
        data_transform: t.Optional[t.Callable] = None,
    ) -> SizedDataLoader:
        """Creates a data loader for the dataset, compatible with the `Trainer` class of the QML Suite.

        Args:
            batch_size (int): number of samples in a batch. Use -1 to use the entire dataset.
            data_transform (t.Optional[t.Callable], optional): transformations to apply to data. Defaults to None.
        """
        # encoded samples must be sequences of integers, so we need to cast them to int to avoid errors
        return new_data_loader(
            self.encoded_samples.astype(np.int_),
            batch_size=batch_size,
            data_transform=data_transform,
        )

    # def resample(self,method):
    #     data_one_hot_encoding = selfies.method()
    #     idx = np.random.choice(len(data_one_hot_encoding), 1000, p=self.probs())
    #     return data_one_hot_encoding[idx]


@dataclass
class CompoundsStatistics:
    unique_compounds: Set[str]  # generated compounds that are unique
    valid_compounds: Set[str]  # generated, unique compounds that are also valid
    unseen_compounds: Set[
        str
    ]  # generated, unique, valid compounds that are also not present in train data
    all_compounds: List[str]
    label_compounds: List[str]
    diversity_fraction: float
    filter_fraction: float
    unique_fraction: float
    # Diversity %
    # Fraction of molecules that pass the filter
    # Fraction of unique molecules

    @property
    def n_unique(self) -> int:
        return len(self.unique_compounds)

    @property
    def n_valid(self) -> int:
        return len(self.valid_compounds)

    @property
    def n_unseen(self) -> int:
        return len(self.unseen_compounds)

    @property
    def total_compounds(self) -> int:
        return len(self.all_compounds)


def compute_compound_stats(
    compounds: Tensor,
    decoder_fn: Callable[[Tensor], List[str]],
    diversity_fn: Callable,
    validity_fn: Callable[[List[str]], List[str]],
    train_compounds: List[str],
) -> CompoundsStatistics:
    generated_compounds = decoder_fn(compounds)

    # truncate samples by removing anything that comes after the `pad_char`
    # generated_compounds = truncate_fn(generated_compounds)
    diversity_fraction = diversity_fn(generated_compounds)

    unqiue_generated_compounds = set(generated_compounds)

    # gives us only valid unique compounds
    filtered_set = validity_fn(generated_compounds)
    unique_valid_compounds = set(filtered_set)

    # valid unique compounds that are also not present in the training data
    unique_train_compounds = set(train_compounds)
    unique_unseen_valid_compounds = unique_valid_compounds.difference(
        unique_train_compounds
    )
    # fraction of unique valid compounds that are unseen
    unique_fraction = 100 * len(unqiue_generated_compounds) / len(compounds)
    filter_fraction = 100 * len(filtered_set) / len(compounds)

    stats = CompoundsStatistics(
        unqiue_generated_compounds,
        unique_valid_compounds,
        unique_unseen_valid_compounds,
        generated_compounds,
        [1] * len(generated_compounds),
        diversity_fraction,
        filter_fraction,
        unique_fraction,
    )

    return stats


def get_binary_property_array(mol_compounds: Sequence[str]) -> torch.Tensor:
    """Returns a tensor where the value at each index is either 1 or 0,
    indicating the presence or absence of a specific molecular property.

    Args:
        mol_compound (Sequence[str]): a collection of molecular compounds in their
            string representation.
    """
    new_data = []
    for compound in mol_compounds:
        try:
            reward_1 = compute_array_value(compound)
            lip = compute_lipinski(compound, mol_weight_ref=800)
            reward = np.append(lip[2], reward_1)
            new_data.append(reward)

        except:
            reward = np.asarray([0, 0, 0, 0, 0])
            new_data.append(reward)

    return torch.Tensor(new_data)

def compute_compound_stats_new(
    generated_compounds,
    diversity_fn: Callable,
    validity_fn: Callable[[List[str]], List[str]],
    train_compounds: List[str],
) -> CompoundsStatistics:

    # truncate samples by removing anything that comes after the `pad_char`
    # generated_compounds = truncate_fn(generated_compounds)
    diversity_fraction = diversity_fn(generated_compounds)

    unqiue_generated_compounds = set(generated_compounds)

    # gives us only valid unique compounds
    filtered_set = validity_fn(generated_compounds)
    unique_valid_compounds = set(filtered_set)

    # valid unique compounds that are also not present in the training data
    unique_train_compounds = set(train_compounds)
    unique_unseen_valid_compounds = unique_valid_compounds.difference(
        unique_train_compounds
    )
    # fraction of unique valid compounds that are unseen
    unique_fraction = 100 * len(unqiue_generated_compounds) / len(generated_compounds)
    filter_fraction = 100 * len(filtered_set) / len(generated_compounds)

    stats = CompoundsStatistics(
        unqiue_generated_compounds,
        unique_valid_compounds,
        unique_unseen_valid_compounds,
        generated_compounds,
        [1] * len(generated_compounds),
        diversity_fraction,
        filter_fraction,
        unique_fraction,
    )

    return stats
