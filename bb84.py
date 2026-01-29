import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def generate_random_bits(n):
    """Generate n random bits (0 or 1)."""
    return np.random.randint(0, 2, n)


def generate_random_bases(n):
    """
    Generate n random bases.
    0 = Computational basis (Z)
    1 = Hadamard basis (X)
    """
    return np.random.randint(0, 2, n)


def encode_qubit(bit, basis):
    """
    Encode a classical bit into a quantum state based on the chosen basis.
    
    Args:
        bit: Classical bit (0 or 1)
        basis: Measurement basis (0 for Z/computational, 1 for X/Hadamard)
    
    Returns:
        QuantumCircuit: Circuit with the encoded qubit
    
    Encoding scheme:
    - Z basis: |0⟩ for bit 0, |1⟩ for bit 1
    - X basis: |+⟩ for bit 0, |-⟩ for bit 1
    """
    qc = QuantumCircuit(1, 1)
    
    # encode the bit gurlll
    if bit == 1:
        qc.x(0)  # now apply X gate to get |1⟩
    
    # if X basis, apply Hadamard to go to |+⟩ or |-⟩ state
    if basis == 1:
        qc.h(0)
    
    return qc


def eve_intercept(qc, eve_basis):
    """
    Eve intercepts and measures the qubit in a random basis,
    then re-encodes and sends to Bob (Intercept-Resend attack).
    
    Args:
        qc: QuantumCircuit containing the qubit to intercept
        eve_basis: Eve's measurement basis (0 for Z, 1 for X)
    
    Returns:
        tuple: (intercepted_circuit, eve_measurement)
    """
    # Eve measures in her chosen basis (go gurl)
    if eve_basis == 1:
        qc.h(0)
    
    qc.measure(0, 0)
    
    # simulate Eve's measurement (come on)
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1, memory=True).result()
    eve_measurement = int(result.get_memory()[0])
    
    # Eve re-encodes the measured bit in the same basis (huh)
    qc_new = QuantumCircuit(1, 1)
    if eve_measurement == 1:
        qc_new.x(0)
    if eve_basis == 1:
        qc_new.h(0)
    
    return qc_new, eve_measurement


def measure_qubit(qc, basis, noise_model=None):
    """
    Measure a qubit in the specified basis with optional noise.
    
    Args:
        qc: QuantumCircuit containing the qubit to measure
        basis: Measurement basis (0 for Z, 1 for X)
        noise_model: Optional noise model to apply
    
    Returns:
        int: Measurement result (0 or 1)
    """
    # if measuring in X basis, apply Hadamard before measurement (yeah)
    if basis == 1:
        qc.h(0)
    
    # now measure in computational basis ya peasant
    qc.measure(0, 0)
    
    # simulate the measurement with optional noise like brrrrrr
    simulator = AerSimulator(noise_model=noise_model)
    transpiled_qc = transpile(qc, simulator)
    result = simulator.run(transpiled_qc, shots=1, memory=True).result()
    measurement = int(result.get_memory()[0])
    
    return measurement


def create_satellite_noise_model(depolarizing_prob=0.01):
    """
    Create a noise model simulating atmospheric interference
    in satellite-to-ground quantum communication.
    
    Args:
        depolarizing_prob: Probability of depolarizing error (default 1%)
    
    Returns:
        NoiseModel: Configured noise model
    """
    noise_model = NoiseModel()
    
    # now depolarizing dat error on single-qubit gates
    error_gate1 = depolarizing_error(depolarizing_prob, 1)
    
    # add errors to all single-qubit gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ['h', 'x'])
    
    # add measurement error
    noise_model.add_all_qubit_quantum_error(error_gate1, ['measure'])
    
    return noise_model


def sift_keys(alice_bits, alice_bases, bob_bases, bob_measurements):
    """
    Perform basis sifting to extract the shared secret key.
    
    Args:
        alice_bits: Alice's original bits
        alice_bases: Alice's encoding bases
        bob_bases: Bob's measurement bases
        bob_measurements: Bob's measurement results
    
    Returns:
        tuple: (alice_key, bob_key, matching_indices)
    """
    alice_key = []
    bob_key = []
    matching_indices = []
    
    for i in range(len(alice_bits)):
        # keep them bits only where alice and bob used the same basis
        if alice_bases[i] == bob_bases[i]:
            alice_key.append(int(alice_bits[i]))
            bob_key.append(int(bob_measurements[i]))
            matching_indices.append(i)
    
    return alice_key, bob_key, matching_indices


def calculate_qber(alice_key, bob_key, sample_size=None):
    """
    Calculate the Quantum Bit Error Rate (QBER) by comparing
    a sample of Alice's and Bob's keys.
    
    Args:
        alice_key: Alice's sifted key
        bob_key: Bob's sifted key
        sample_size: Number of bits to compare (None = use all)
    
    Returns:
        tuple: (qber, errors, total_checked)
    """
    if sample_size is None or sample_size > len(alice_key):
        sample_size = len(alice_key)
    
    # in real QKD, a random sample would be chosen and announced publicly (lottery shi)
    # for simulation, we check all bits (duh)
    errors = sum(1 for i in range(sample_size) if alice_key[i] != bob_key[i])
    qber = (errors / sample_size) * 100 if sample_size > 0 else 0
    
    return qber, errors, sample_size


def main():
    """Main function to run the BB84 protocol simulation with Eve and noise."""
    
    print("=" * 70)
    print("BB84 QKD with Eavesdropping and Satellite Noise Simulation")
    print("=" * 70)
    print()
    
    # configuration
    n_bits = 200  
    eve_present = True  
    noise_enabled = True  
    depolarizing_prob = 0.01  
    qber_threshold = 11.0  
    
    print("Configuration:")
    print(f"  - Number of qubits: {n_bits}")
    print(f"  - Eve (eavesdropper): {'ACTIVE' if eve_present else 'INACTIVE'}")
    print(f"  - Atmospheric noise: {'ENABLED' if noise_enabled else 'DISABLED'}")
    if noise_enabled:
        print(f"  - Depolarizing probability: {depolarizing_prob*100:.1f}%")
    print(f"  - QBER threshold: {qber_threshold}%")
    print()
    
    # now we create that noise model if enabled
    noise_model = create_satellite_noise_model(depolarizing_prob) if noise_enabled else None
    
    # step 1: alice prepares qubits (qween)
    alice_bits = generate_random_bits(n_bits)
    alice_bases = generate_random_bases(n_bits)
    
    print("=" * 70)
    print("STEP 1: Alice prepares qubits")
    print("=" * 70)
    print(f"Alice's bits (first 20): {alice_bits[:20]}")
    print(f"Alice's bases (first 20): {alice_bases[:20]}")
    print()
    
    # step 2: quantum transmission (with possible Eve interception)
    print("=" * 70)
    print("step 2: quantum transmission")
    print("=" * 70)
    
    if eve_present:
        print("Eve is intercepting qubits")
        eve_bases = generate_random_bases(n_bits)
        eve_measurements = []
        transmitted_circuits = []
        
        for i in range(n_bits):
            # alice encodes
            qc = encode_qubit(alice_bits[i], alice_bases[i])
            
            # Eve intercepts and resends
            qc_resent, eve_measurement = eve_intercept(qc, eve_bases[i])
            eve_measurements.append(eve_measurement)
            transmitted_circuits.append(qc_resent)
        
        print(f"Eve's bases (first 20): {eve_bases[:20]}")
        print(f"Eve's measurements (first 20): {eve_measurements[:20]}")
    else:
        print(" Direct quantum channel (no eavesdropping)")
        transmitted_circuits = [encode_qubit(alice_bits[i], alice_bases[i]) 
                                for i in range(n_bits)]
    
    print()
    
    # Step 3: Bob measures (men always be late)
    print("=" * 70)
    print("STEP 3: Bob measures qubits")
    print("=" * 70)
    
    bob_bases = generate_random_bases(n_bits)
    bob_measurements = []
    
    for i in range(n_bits):
        qc = transmitted_circuits[i]
        measurement = measure_qubit(qc, bob_bases[i], noise_model)
        bob_measurements.append(measurement)
    
    print(f"Bob's bases (first 20): {bob_bases[:20]}")
    print(f"Bob's measurements (first 20): {bob_measurements[:20]}")
    print()
    
    # Step 4: Sifting
    print("=" * 70)
    print("STEP 4: Basis sifting (classical channel)")
    print("=" * 70)
    
    alice_key, bob_key, matching_indices = sift_keys(
        alice_bits, alice_bases, bob_bases, bob_measurements
    )
    
    print(f"Bases matched: {len(matching_indices)} out of {n_bits}")
    print(f"Sifting efficiency: {len(matching_indices)/n_bits*100:.2f}%")
    print()
    
    # Step 5: QBER calculation
    print("=" * 70)
    print("step 5: Quantum Bit Error Rate (QBER) Analysis")
    print("=" * 70)
    
    qber, errors, total_checked = calculate_qber(alice_key, bob_key)
    
    print(f"Bits compared: {total_checked}")
    print(f"Errors detected: {errors}")
    print(f"QBER: {qber:.2f}%")
    print(f"Threshold: {qber_threshold}%")
    print()
    
    # Step 6: Security decision
    print("=" * 70)
    print("STEP 6: Security Assessment")
    print("=" * 70)
    
    if qber > qber_threshold:
        print(f"protocol aborted")
        print(f"QBER ({qber:.2f}%) exceeds threshold ({qber_threshold}%)")
        print()
        print("Possible causes:")
        print("  1. Eavesdropping detected (Eve's intercept-resend attack)")
        print("  2. Excessive channel noise")
        print("  3. Equipment malfunction")
        print()
        print("Recommendation: Do NOT use this key for encryption!")
        print("Action: Retry key distribution or switch to backup channel.")
    else:
        print(f"protocol successful")
        print(f"QBER ({qber:.2f}%) is below threshold ({qber_threshold}%)")
        print()
        print("The quantum channel is secure enough to proceed.")
        print("Final key length after error correction: ~", len(alice_key) - errors, "bits")
        print()
        
        # Display partial key (in real QKD, this would never be shown)
        print("Shared Secret Key (first 50 bits):")
        key_string = ''.join(map(str, alice_key[:50]))
        print(f"  {key_string}")
    
    print()
    print("=" * 70)
    print("statistics summary")
    print("=" * 70)
    print(f"Total qubits transmitted: {n_bits}")
    print(f"Qubits after sifting: {len(alice_key)}")
    print(f"Sifting efficiency: {len(alice_key)/n_bits*100:.2f}%")
    print(f"Bit errors: {errors}")
    print(f"QBER: {qber:.2f}%")
    print(f"Protocol status: {'ABORTED' if qber > qber_threshold else 'SUCCESS'}")
    
    # Theoretical expectations
    print()
    print("Theoretical Analysis:")
    print(f"  - Expected sifting rate: ~50%")
    print(f"  - Expected QBER (no Eve, no noise): ~0%")
    if eve_present:
        print(f"  - Expected QBER (with Eve): ~25%")
        print(f"    (Eve guesses wrong basis 50% of time, causing 50% errors in those cases)")
    if noise_enabled:
        print(f"  - Additional QBER from noise: ~{depolarizing_prob*100:.1f}%")
    
    print()
    print("=" * 70)
    print("Simulation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    # set random seed for reproducibility (optional - comment out for true randomness)
    np.random.seed(42)
    
    main()