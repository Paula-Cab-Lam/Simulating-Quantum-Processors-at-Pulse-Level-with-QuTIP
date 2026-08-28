#Packages needed for the quantum simulation
import numpy as np
import json
from qutip import basis, Qobj, expect
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits
from qutip_qip.compiler import SCQubitsCompiler


class SCQubitsCompilerNoDRAG(SCQubitsCompiler):
    """Compilador SCQubits without DRAG correction"""

    def __init__(self, num_qubits, params):
        super().__init__(num_qubits, params)
        #Deactivate DRAG
        self.args["DRAG"] = False


def load_valencia(file_name = 'props_valencia.json'):
    """Open de json file from fake_valencia and get the information about the processor"""
    with open(file_name, 'r') as f:
        valencia_data = json.load(f)

    #Data to store from the qubits
    T1 = []
    T2 = []
    frequency = []
    alpha = []

    for _, qubit in enumerate(valencia_data["qubits"]):
        #For t1 and t2 we need to transform its units
        T1.append(qubit[0]['value']*1e3)   
        T2.append(qubit[1]['value']*1e3)   
        frequency.append(qubit[2]['value'])
        alpha.append(qubit[3]['value'])

    return T1, T2, frequency, alpha


def create_circuit(gate: str):
    """X circuit or RX circuit"""
    circuit = QubitCircuit(1)
    circuit.add_gate(gate, targets = 0)
    return circuit


def sample(num_trials: int, fin_result_state: Qobj):
    #Find the final probabilities
    p0 = expect(basis(3, 0)*basis(3, 0).dag(), fin_result_state)
    p1 = expect(basis(3, 1)*basis(3, 1).dag(), fin_result_state) + expect(basis(3, 2)*basis(3, 2).dag(), fin_result_state) #We add the leakage to state 1

    #Sample num_trials
    total = p0 + p1
    sample = np.random.multinomial(num_trials, [p0/total, p1/total])
    return sample/num_trials


def set_amplitude(processor, original_pulses: dict, scale: float):
    for pulse in processor.pulses:
        if pulse.label not in original_pulses:
            continue
        if pulse.label.startswith("sz"):
            pulse.coeff = original_pulses[pulse.label] * scale * scale   #sigma_z scales as amplitude squared for DRAG
        else:
            pulse.coeff = original_pulses[pulse.label] * scale           #sigma_x and sigma_y scale directly


def build_processor(circuit, qubit = 0, drag = False):
    """Define the processor for one qubit of fake_valencia and load the circuit"""
    T1, T2, frequency, alpha = load_valencia()

    #Time of a single qubit gate
    t_single = 35.5555555556
    omega_single = 1/t_single

    processor = SCQubits(num_qubits = 1, dims = [3], wq = [frequency[qubit]], alpha = [alpha[qubit]], omega_single = [omega_single], t1 = T1[qubit], t2 = T2[qubit])

    if drag:
        compiler = SCQubitsCompiler(num_qubits = 1, params = processor.params)
    else:
        compiler = SCQubitsCompilerNoDRAG(num_qubits = 1, params = processor.params)

    processor.load_circuit(qc = circuit, schedule_mode = 'ASAP', compiler = compiler)

    #Store the pulses before scaling them
    original_pulses = {}
    for pulse in processor.pulses:
        if pulse.coeff is not None:
            original_pulses[pulse.label] = pulse.coeff.copy()

    return processor, original_pulses


def ideal_distribution(gate: str):
    """Ideal probabilities (p0, p1) for the gate"""
    if gate == 'X':
        return 0.0, 1.0
    else:
        return 0.5, 0.5


def simulate_amplitude(processor, original_pulses: dict, amplitude: float, gate: str, num_trials: int = 1024):
    """Simulate one amplitude and return the metric and the exact populations"""
    initial_state = basis(3, 0)
    set_amplitude(processor = processor, original_pulses = original_pulses, scale = amplitude)
    result = processor.run_state(init_state = initial_state)
    fin_state = result.states[-1]

    #Exact populations (no shot noise) to compare later
    #p1_exact = expect(basis(3, 1)*basis(3, 1).dag(), fin_state) + expect(basis(3, 2)*basis(3, 2).dag(), fin_state)

    #Sample and find probability distribution
    p0, p1 = sample(num_trials = num_trials, fin_result_state = fin_state)
    #Compute the metric
    p0_ideal, p1_ideal = ideal_distribution(gate)
    fid = (np.sqrt(p0)*np.sqrt(p0_ideal) + np.sqrt(p1)*np.sqrt(p1_ideal))**2

    return float(fid)


def run_scan(gate: str, num_simulations: int = 400, amp_max: float = 5.9, num_trials: int = 1024,
     drag = False, file_name = None, plot_every = None, plot_folder = 'pulse_plots'):
    """Simulate num_simulations random amplitudes and save (amplitude, fidelity)"""
    import matplotlib.pyplot as plt
    import os

    circuit = create_circuit(gate)
    processor, original_pulses = build_processor(circuit, drag = drag)

    if plot_every is not None:
        os.makedirs(plot_folder, exist_ok = True)

    amplitudes = np.random.uniform(0, amp_max, num_simulations)
    amp_fidelity = []

    for counter, amp in enumerate(amplitudes):
        fid = simulate_amplitude(processor, original_pulses, amp, gate, num_trials)
        amp_fidelity.append((float(amp), fid))

        #Save the pulse plot for some amplitudes
        if plot_every is not None and counter % plot_every == 0:
            fig, ax = processor.plot_pulses(title = f'Control pulse of SCQubit, amp = {amp:.3f}', figsize = (8, 4), dpi = 100, rescale_pulse_coeffs = False)
            fig.savefig(f'{plot_folder}/pulse_amp_{amp:.3f}.png', dpi = 100, bbox_inches = 'tight')
            plt.close(fig)

    if file_name is not None:
        with open(file_name, 'w') as f:
            json.dump(amp_fidelity, f, indent = 2)

    return np.array(amp_fidelity)
