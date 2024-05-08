import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.providers.aer import noise
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from tqdm import tqdm
from qiskit import IBMQ
import numpy as np
import seaborn as sns
from matplotlib.pylab import plt
import time

# import torch.nn.functional as F
import typing
from scipy import optimize
from qiskit import execute
from .error import ErrorMitigation
# %%
epsilon = 1e-5


class ExactNLL:
    def measure_distance(self, true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
        return -np.dot(
            true_probs, np.log(np.clip(pred_probs, a_min=epsilon, a_max=None)),
        )


class QCBMGenerator:
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        n_shots: int,
        qcbm_circuit,
        train_backend=None,
        train_error_mitigation = False,
        sampeling_backend=None,
        sampeling_error_mitigation=False,
        optimizer="Powell",
        lr=0.001,
        initial_params=None,
        param_initializer: typing.Callable[
            [int], np.ndarray
        ] = lambda n_params: np.random.uniform(-np.pi / 2, np.pi / 2, size=n_params),
        distance_fn=ExactNLL(),
        lr_step: int = 1,
        xtol: float = 1e-6,
        ftol: float = 1e-6,
        lower_bounds=-np.pi / 2,
        upper_bounds=np.pi / 2,
        device="cpu",
        max_shots=20000,
        scale_factors=[11]
        
    ):  
        self._max_cost= 1000
        self.n_call = 0
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.n_params = qcbm_circuit.num_parameters
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.lr_step = lr_step
        self.xtol = xtol
        self.ftol = ftol
        self.all_loss_values = []
        self.lower_bounds = np.full(self.n_params, lower_bounds)
        self.upper_bounds = np.full(self.n_params, upper_bounds)
        self.max_shots = max_shots
        self.scale_factors = scale_factors
        if initial_params is not None:
            self.initial_params = initial_params
        else:
            self.initial_params = param_initializer(
                self.n_params
            )  # self.get_initial_param()
        self.optimized_params = self.initial_params
        self.qcbm_circuit = qcbm_circuit
        self.loss = []
        if train_backend:
            self.train_backend = train_backend
            self.error_mitigation_train = ErrorMitigation(self.train_backend)
            self.train_error_mitigation = train_error_mitigation
        else:
            self.train_backend = Aer.get_backend("statevector_simulator")
        
        if sampeling_backend:
            self.sampeling_backend = sampeling_backend
            self.error_mitigation_sampeling = ErrorMitigation(self.sampeling_backend)
            self.sampeling_error_mitigation = sampeling_error_mitigation
        else:
            self.sampeling_backend = Aer.get_backend("statevector_simulator")
            
        self.cost_function = distance_fn.measure_distance


    def objective_function(self, theta_values):
        backend =self.train_backend
        qc = self.qcbm_circuit.copy()
        qc.measure_all()
        # print(self.n_call)
        param_dict = {
            qc.parameters[i]: theta_values[i] for i in range(len(theta_values))
        }
        qc = qc.bind_parameters(param_dict)
        qobj = assemble(qc, shots=self.n_shots)
        # result = self.backend.run(qobj, noise_model=self.noise_model).result()
        # counts = result.get_counts()
        t_circuit = transpile(qc, backend)

        # Execute the transpiled circuit
        # error_mitigation.run_pipeline(circuit_16, scale_factors,randomize=False)
        try:
            if self.train_error_mitigation:
                counts = self.error_mitigation_train.run_pipeline(t_circuit,self.scale_factors,shots=self.n_shots)[0]
            else:
                job = execute(
                    t_circuit, backend, shots=self.n_shots
                )
                result = job.result()
                counts = result.get_counts()
            # Create a NumPy array to store the empirical distribution
            Q = np.zeros(len(self.input_distribution), dtype=np.float64)

            # Fill the empirical distribution array using the counts
            for idx, key in enumerate(self.input_distribution.keys()):
                Q[idx] = counts.get("".join(map(str, key)), 0) / self.n_shots

            # Add a small constant to avoid division by zero in the cost function
            Q += 1e-15

            P = self.P.copy()

            cost = self.cost_function(P, Q)
            self.n_call += 1
        except:
            cost=self._max_cost
            print("IBMQ ERROR TRAINING")
        return cost

    def get_initial_param(self):
        initial_params = np.random.rand(self.n_params) * np.pi * 0.5
        return np.array(initial_params)

    def train(self, epochs, bitstrings, probabilities,warm_started = False):
        self.epochs = epochs
        self.input_distribution = dict(zip(map(tuple, bitstrings), probabilities))
        self.P = np.array(list(self.input_distribution.values()), dtype=np.float64)
        if warm_started:
            param = np.array(self.initial_params, dtype=np.float64)
        else:
            self.initial_params = self.get_initial_param()
            param = np.array(self.initial_params, dtype=np.float64)
            self.loss = []

        history = {"params": [], "loss": []}

        def loss_fn(param, cost):
            # cost = self.objective_function(x)
            history["params"].append(param.copy())
            history["loss"].append(cost)
            # print(fn)

        progress_bar = tqdm(range(epochs), desc="Training progress")
        bounds = optimize.Bounds(self.lower_bounds, self.upper_bounds)

        # Initialize optimizer and scheduler
        loss = []
        params = []
        for _ in progress_bar:
            result = optimize.minimize(
                self.objective_function,
                param.copy(),
                method=self.optimizer,
                bounds=bounds,
                options={
                    "maxiter": self.lr_step,
                    "xtol": self.xtol,
                    "ftol": self.ftol,
                },
            )
            param = result.x
            cost = result.fun
            params.append(param)
            loss.append(cost)
            loss_fn(param, cost)
            progress_bar.set_postfix({"cost": cost,"warm_started":warm_started})
        idx = np.argmin(loss)
        self.loss += history["loss"]
        self.optimized_params = params[idx]
        self.history = history
        # prior_train_cache["history"]["opt_value"]
        re = {"history": {"opt_value": self.loss[::-1], "loss": self.loss}}
        return re

    def generate(self, num_samples):
        samples_per_call = self.max_shots
        binary_samples = []
        circuit = self.qcbm_circuit.copy()
        circuit.measure_all()
        param_dict = {
            circuit.parameters[i]: self.optimized_params[i]
            for i in range(len(self.optimized_params))
        }
        circuit = circuit.assign_parameters(param_dict)
        # Transpile the circuit
        backend = self.sampeling_backend
        t_circuit = transpile(circuit, backend)
        while num_samples > 0:
            current_samples = min(samples_per_call, num_samples)
            # Create a parameter dictionary from the optimized parameters
            # Copy the QCBM circuit and assign the parameter

            # Execute the transpiled circuit
            try:
                if self.sampeling_error_mitigation:
                    counts = self.error_mitigation_sampeling.run_pipeline(t_circuit,self.scale_factors,shots=current_samples)[0]
                else:
                    job = execute(
                        t_circuit, backend, shots=current_samples
                    )
                    
                    result = job.result()

                    # Process the result
                    counts = result.get_counts()
                for bitstring, count in counts.items():
                    for _ in range(count):
                        binary_samples.append([int(bit) for bit in bitstring])

                num_samples -= current_samples
            except:
                print("IBMQ ERROR SAMPELING")

        return np.array(binary_samples, dtype=int)

class RandomCardinalityGenerator:
    def __init__(self, size, cardinality):
        self.size = size
        self.cardinality = cardinality

    def generate(self, n_samples):
        cardinality = self.cardinality

        if cardinality > self.size:
            raise ValueError(
                "Cardinality should be less than or equal to the bitstring length."
            )

        bitstrings = np.zeros((n_samples, self.size), dtype=np.int32)

        for i in range(n_samples):
            indices = np.random.permutation(self.size)[:cardinality]
            bitstrings[i, indices] = 1

        return bitstrings

# %%
