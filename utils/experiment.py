import json
import os
import random
import string
import typing as t
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from orquestra.qml.api import TrainCache


class Experiment:
    _figures_dir = "figures"
    _saved_models_dir = "saved_models"
    _samples_dir = "samples"
    _compound_dir = "compound_samples"

    def __init__(
        self, run_id: str, root_path: Path = Path("experiment_results").resolve()
    ) -> None:
        self._run_id = run_id
        self._path = root_path / self.run_id
        self._user_properties: Dict[Any, Any] = {}

        self.path_to_dataset: Optional[str] = None
        self.dataset_id: Optional[str] = None
        self.n_epochs: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.model_configurations: List[Dict] = []
        self.train_cache: Optional[TrainCache] = None
        self.seed: Optional[int] = None

        self._base_directories = [
            self._figures_dir,
            self._saved_models_dir,
            self._samples_dir,
            self._compound_dir
        ]

        self._registered_directories: Dict[str, Path] = {}

    def __repr__(self) -> str:
        return f"Experiment(run_id={self.run_id}, path={self.path})"

    def __getitem__(self, name: Any) -> Any:
        return self.user_properties[name]

    def __setitem__(self, name: Any, value: Any):
        self._user_properties[name] = value

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def user_properties(self) -> Dict:
        return self._user_properties

    @property
    def registered_directories(self) -> Dict[str, Path]:
        """Returns a dictionary, where keys are names of directories registered with the experiment,
        and the values are their absolute paths.
        """
        return self._registered_directories

    @property
    def saved_models_dir(self) -> Path:
        return self._abs_path(self._saved_models_dir)

    @property
    def figures_directory(self) -> Path:
        return self._abs_path(self._figures_dir)

    def _abs_path(self, path: Union[Path, str]) -> Path:
        return self._path / path

    def _make_experiment_dirs(self):
        dirs = chain(self._base_directories, self._registered_directories.keys())
        for dir in dirs:
            self.create_directory(self._abs_path(dir))

    def start(self) -> None:
        self._make_experiment_dirs()

    def set(self, **kwargs) -> "Experiment":
        self._user_properties.update(kwargs)

    def get(self, key: Any) -> Any:
        return self.__getitem__(key)

    def create_directory(self, path: Path) -> Path:
        """Creates a directory inside of the experiment's root directory.
        Returns the absolute path to the newly created directory.

        Args:
            path (Path): path of directory relative to that of experiment root directory.
        """
        abs_path = self._abs_path(path)
        if abs_path.exists() is False:
            os.makedirs(str(abs_path))

        return abs_path

    def register_directory(self, path: Path) -> Path:
        """Registers a directory with the experiment, and will be created when
        `Experiment.start` is called. Returns the absolute path of the registered directory.

        Args:
            path (Path): path of directory relative to that of experiment root directory.
        """
        self._registered_directories[str(path)] = self._abs_path(path)
        return self._abs_path(path)

    def as_dict(self) -> Dict:
        return dict(
            run_id=self.run_id,
            path=str(self.path),
            path_to_dataset=str(self.path_to_dataset),
            dataset_id=self.dataset_id,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            model_configurations=self.model_configurations,
            train_cache=self.train_cache.__dict__,
            seed=self.seed,
            user_properties=self.user_properties,
        )

    def save_json(self):
        save_path = self._abs_path("experiment.json")
        d = self.as_dict()
        d["train_cache"].pop("_reserved_keys", None)
        d["train_cache"].pop("logger", None)
        d["train_cache"].pop("_train_batch", None)

        with open(str(save_path), "w") as f:
            json.dump(self.as_dict(), f)


class LegacyExperiment:
    _base_directories = ["figures", "saved_models", "compound_samples"]

    def __init__(
        self,
        run_id: str,
        root_dir: Path = Path("experiment_results").resolve(),
        results: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> None:
        self.path_to_dataset: str
        self.dataset_id: str
        self.n_epochs: int
        self.batch_size: int
        self.model_configurations: List
        self.train_cache: TrainCache

        self._root_dir = Path(root_dir)
        self._dt_format = "%d%b%YT%H%M"
        self._experiments_directory = root_dir
        self._results = results if results is not None else {}
        self.run_id = run_id if run_id is not None else self._generate_id()
        self.date = self.results.get("date") or datetime.today().strftime(
            self._dt_format
        )
        self.notes = ""

        # this won't change anything if experiment was loaded from pre-existing file
        self.update_results(dict(run_id=self.run_id, date=self.date))
        self._base_directories.append("epoch_plots/{}".format(self.run_id))

        self._make_experiment_dirs()

    def _make_experiment_dirs(self):
        experiment_root = self._experiments_directory
        for dir in self._base_directories:
            new_dir = experiment_root / dir
            if not new_dir.exists():
                os.makedirs(str(new_dir))

    def _generate_id(self) -> str:
        """Generates an ID for the experiment."""
        return "".join([random.choice(string.ascii_letters) for _ in range(10)])

    @classmethod
    def from_file(cls, filepath: str) -> "Experiment":
        with open(filepath, "r+") as f:
            results = json.load(f)
        return cls(results=results)

    def path_to_model_weights(self, path_to_dir: str) -> str:
        """Returns the path to the weights of the model trained during the current experiment."""

        # first try the .pt file
        file = os.path.join(path_to_dir, f"{self.run_id}.pt")

        if os.path.isfile(file) is False:
            file = file.replace(".pt", ".pth")

        # check if .pth file exists
        if os.path.isfile(file) is False:
            raise RuntimeError("Unable to find file with model weights.")

        return file

    @property
    def results(self) -> t.Dict[str, t.Any]:
        return self._results

    def update_results(self, dict: t.Dict[str, t.Any]):
        """Updates the results dictionary."""
        self._results.update(dict)

    def add_note(self, note: str):
        self.notes = f"{self.notes}\n{note}"

    def save_experiment(self, filename: str):
        p = Path(self._experiments_directory)
        if p.exists() is False:
            p.mkdir()

        self.results.update(dict(notes=self.notes))

        path_to_file = os.path.join(self._experiments_directory, filename)
        with open(path_to_file, "w+") as f:
            json.dump(self._results, f)

    def update(self, **kwargs: Any) -> "Experiment":
        self.update_results(kwargs)
        return self
