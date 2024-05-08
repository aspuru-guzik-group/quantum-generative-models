import typing as t
from ast import Call

from orquestra.qml.api import Callback, GenerativeModel, TrainResult, convert_to_numpy

from .chem import SmilesEncoding


class DisplaySmilesCallback(Callback):
    _repr_fields: t.List[str] = ["frequency", "n_smiles"]

    def __init__(
        self, smiles_encoding: SmilesEncoding, frequency: int = 1, n_smiles: int = 10
    ) -> None:
        """Callback that displays generated smiles after a specified number of training epochs.

        Note:
            Currently only instances of GenerativeModel are supported.

        Args:
            smiles_encoding (SmilesEncoding): class that contains Smiles dataset and encodings.
            frequency (int, optional): number of epochs between successive plots. Defaults to 1.
            n_smiles (int, optional): number of smiles to display.
        """
        super().__init__()
        self.smiles_encoding = smiles_encoding
        self.frequency = frequency
        self.n_smiles = n_smiles

    def set_model(self, model: GenerativeModel) -> None:
        if not isinstance(model, GenerativeModel):
            raise RuntimeError(
                "The PlotSamples currently only supports instances of `GenerativeModel`."
            )
        return super().set_model(model)

    def on_epoch_end(self, epoch: int, checkpoint_cache: TrainResult):
        if epoch % self.frequency != 0:
            return

        # TODO[@djvaroli]: enable SupervisedModels to be used here (e.g. conditional GAN)
        decoded_smiles = self.smiles_encoding.decode_smiles(
            self._model.generate(self.n_smiles)
        )
        print(decoded_smiles)

    def modify_cache(self, checkpoint_cache: TrainResult):
        pass
