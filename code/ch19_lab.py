"""Compare mixed-state marginals and a partial-transpose threshold."""
import numpy as np


def partial_transpose(state):
    return state.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


def marginal(state):
    return np.trace(state.reshape(2, 2, 2, 2), axis1=1, axis2=3)


singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
singlet_rho = np.outer(singlet, singlet)
correlated = np.diag([0.5, 0, 0, 0.5])
assert np.allclose(marginal(singlet_rho), marginal(correlated))
assert np.allclose(np.linalg.eigvalsh(partial_transpose(correlated)),
                   [0, 0, 0.5, 0.5])
for p in np.linspace(0, 1, 61):
    state = p * singlet_rho + (1 - p) * np.eye(4) / 4
    eigenvalues = np.linalg.eigvalsh(partial_transpose(state))
    expected = np.array([(1 - 3 * p) / 4] + [(1 + p) / 4] * 3)
    assert np.allclose(eigenvalues, expected)
print("Equal local marginals for Bell and correlated mixture: True")
for p in [0, 1 / 3, 0.5, 1]:
    value = (1 - 3 * p) / 4
    print(f"p={p:.6f}: minimum partial-transpose eigenvalue={value:.6f}")
print("Sixty-one-point spectrum sweep passed: True")

# Reconstruct the zero-field Heisenberg thermal state from its Hamiltonian.
x = np.array([[0, 1], [1, 0]], complex)
y = np.array([[0, -1j], [1j, 0]])
z = np.diag([1, -1])
coupling = 1.0  # J fixes the energy unit; beta is in inverse energy units.
hamiltonian = sum(np.kron(q, q) for q in [x, y, z]) * coupling / 4
energy, vectors = np.linalg.eigh(hamiltonian)
for beta in np.linspace(0, 4, 21):
    weights = np.exp(-beta * (energy - energy.min()))
    thermal = (vectors * (weights / weights.sum())) @ vectors.conj().T
    exp_beta_j = np.exp(beta * coupling)
    p = (exp_beta_j - 1) / (exp_beta_j + 3)
    assert np.allclose(thermal, p * singlet_rho + (1 - p) * np.eye(4) / 4)

boundary = np.zeros((4, 4), complex)
for axis in [x, y, z]:
    _, spin_vectors = np.linalg.eigh(axis)
    minus, plus = spin_vectors[:, 0], spin_vectors[:, 1]
    for pair in [np.kron(plus, minus), np.kron(minus, plus)]:
        boundary += np.outer(pair, pair.conj()) / 6
assert np.allclose(boundary, singlet_rho / 3 + np.eye(4) / 6)
print("Heisenberg thermal family and six-product boundary passed: True")
