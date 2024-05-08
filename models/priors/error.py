
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.providers.aer import QasmSimulator
import random
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit import IBMQ




def apply_dynamical_decoupling(circuit, num_pulses):
    qubits = range(circuit.num_qubits)
    for _ in range(num_pulses):
        circuit.barrier(qubits)
        for qubit in qubits:
            circuit.x(qubit)
        circuit.barrier(qubits)
    return circuit


def randomize_compiling(circuit):
    randomized_circuit = circuit.copy()
    gates = [XGate(), YGate(), ZGate()]
    
    for qubit in range(circuit.num_qubits):
        for gate, qargs, cargs in circuit.data:
            pre_gate = random.choice(gates)
            post_gate = random.choice(gates).inverse()
            
            randomized_circuit.append(pre_gate, [qubit])
            randomized_circuit.append(gate, qargs, cargs)
            randomized_circuit.append(post_gate, [qubit])
    
    return randomized_circuit


class ErrorMitigation:
    def __init__(self, backend):
        self.backend = backend
        self.meas_fitter = None

    def richardson_extrapolation(self, circuit, scale_factors, shots=8192, randomize=True):
        mitigated_results = []
        for scale in scale_factors:
            if randomize:
                randomized_circuit = randomize_compiling(circuit)
                scaled_circuit = randomized_circuit.copy()
            else:
                scaled_circuit = circuit.copy()
            
            apply_dynamical_decoupling(scaled_circuit, scale)
            result = execute(scaled_circuit, self.backend, shots=shots).result()
            mitigated_results.append(result.get_counts())
        return mitigated_results



    def calibrate_measurement_error(self, circuit, shots=8192):
        cal_circuits, state_labels = complete_meas_cal(qr=circuit.qregs[0], circlabel='measurement_calibration')
        cal_results = execute(cal_circuits, self.backend, shots=shots).result()
        self.meas_fitter = CompleteMeasFitter(cal_results, state_labels)

    def apply_measurement_error_mitigation(self, raw_results):
        if self.meas_fitter is None:
            raise ValueError("Measurement error calibration has not been performed.")
        mitigated_results = []
        for result in raw_results:
            mitigated_result = self.meas_fitter.filter.apply(result)
            mitigated_results.append(mitigated_result)
        return mitigated_results

    def run_pipeline(self, circuit, scale_factors, shots=8192, randomize=False):
        self.calibrate_measurement_error(circuit, shots)
        raw_results = self.richardson_extrapolation(circuit, scale_factors, shots, randomize)
        mitigated_results = self.apply_measurement_error_mitigation(raw_results)
        return mitigated_results
