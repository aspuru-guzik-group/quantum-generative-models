import typing as t
from collections import deque
import numpy as np
import pandas as pd
import selfies as sf
from orquestra.qml.api import Tensor, convert_to_numpy
from PIL import Image
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

from .mol_methods import *



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
