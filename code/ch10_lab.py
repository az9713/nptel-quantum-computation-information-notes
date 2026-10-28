"""Dense-code decoding and every teleportation branch for complex inputs."""
import itertools
import numpy as np

identity = np.eye(2, dtype=complex)
x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
z_gate = np.diag([1, -1]).astype(complex)
hadamard = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
prepare = cnot @ np.kron(hadamard, identity)
decode = prepare.conj().T
bell = prepare[:, 0]
labels = list(itertools.product(range(2), repeat=2))
bell_basis = {}
for phase, flip in labels:
    index = 2*phase + flip
    vector = prepare[:, index]
    operation = np.linalg.matrix_power(z_gate, phase)
    operation = operation @ np.linalg.matrix_power(x_gate, flip)
    assert np.allclose(np.kron(operation, identity) @ bell, vector)
    assert np.allclose(decode @ vector, np.eye(4)[:, index])
    bell_basis[(phase, flip)] = vector
print("dense coding: 00->00, 01->01, 10->10, 11->11")
print("Bell codeword Gram matrix equals I_4")
assert np.allclose(prepare.conj().T @ prepare, np.eye(4))

inputs = [
    np.array([1, 0], dtype=complex),
    np.array([1, 1], dtype=complex)/np.sqrt(2),
    np.array([np.sqrt(0.3), 1j*np.sqrt(0.7)], dtype=complex),
]
for case, state in enumerate(inputs, 1):
    joint = np.kron(state, bell).reshape(4, 2)
    averaged = np.zeros((2, 2), dtype=complex)
    probabilities, fidelities = [], []
    for phase, flip in labels:
        branch = bell_basis[(phase, flip)].conj() @ joint
        probability = float(np.vdot(branch, branch).real)
        assert np.isclose(probability, 0.25)
        averaged += np.outer(branch, branch.conj())
        correction = np.linalg.matrix_power(z_gate, phase)
        correction = correction @ np.linalg.matrix_power(x_gate, flip)
        restored = correction @ branch/np.sqrt(probability)
        fidelity = float(abs(np.vdot(state, restored))**2)
        assert np.isclose(fidelity, 1)
        probabilities.append(probability)
        fidelities.append(fidelity)
    assert np.allclose(averaged, identity/2)
    print(f"input {case}: branch probabilities="
          + ",".join(f"{p:.2f}" for p in probabilities))
    print("  corrected fidelities="
          + ",".join(f"{f:.6f}" for f in fidelities))
    print("  unread receiver state=I/2")

for phase, flip in labels:
    columns = []
    for basis_state in np.eye(2, dtype=complex):
        joint = np.kron(basis_state, bell).reshape(4, 2)
        columns.append(bell_basis[(phase, flip)].conj() @ joint)
    branch_map = np.column_stack(columns)
    correction = (np.linalg.matrix_power(z_gate, phase)
                  @ np.linalg.matrix_power(x_gate, flip))
    assert np.allclose(correction @ branch_map, identity/2)
print("corrected branch operators K=I/2; sum K rho K^dagger=rho")
