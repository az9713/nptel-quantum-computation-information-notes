"""Evaluate two-qubit concurrence using Hermitian matrix square roots."""
import numpy as np

y = np.array([[0, -1j], [1j, 0]])
flip = np.kron(y, y)
tolerance = 1e-10


def positive_sqrt(matrix):
    assert np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=0)
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues.min() < -tolerance:
        raise ValueError("Matrix is not positive within the stated tolerance")
    return (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.conj().T


def concurrence(state):
    assert np.isclose(np.trace(state), 1, atol=tolerance, rtol=0)
    root = positive_sqrt(state)
    flipped = flip @ state.conj() @ flip
    middle = root @ flipped @ root
    assert np.allclose(middle, middle.conj().T, atol=tolerance, rtol=0)
    middle = (middle + middle.conj().T) / 2
    spectrum = np.linalg.eigvalsh(positive_sqrt(middle))[::-1]
    return max(0.0, spectrum[0] - spectrum[1:].sum())


def binary_entropy(p):
    return max(0.0, -sum(q * np.log2(q) for q in [p, 1 - p] if q > 0))


singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
singlet_rho = np.outer(singlet, singlet)
assert np.isclose(concurrence(singlet_rho), 1, atol=1e-7)
assert np.isclose(concurrence(np.diag([1, 0, 0, 0])), 0)
assert np.isclose(concurrence(np.eye(4) / 4), 0)
for p in np.linspace(0, 1, 41):
    state = p * singlet_rho + (1 - p) * np.eye(4) / 4
    assert np.isclose(concurrence(state), max(0, (3 * p - 1) / 2),
                      atol=1e-7)
for p in [0, 1 / 3, 0.5, 1]:
    value = max(0, (3 * p - 1) / 2)
    entropy = binary_entropy((1 + np.sqrt(1 - value**2)) / 2)
    print(f"p={p:.6f}: C={value:.6f}; E_F={entropy:.6f} bits")
print("Forty-one-point concurrence sweep passed: True")

# Complex pure vectors expose missing conjugations that real Bell states hide.
random = np.random.default_rng(20261028)
for _ in range(24):
    vector = random.normal(size=4) + 1j * random.normal(size=4)
    vector /= np.linalg.norm(vector)
    coefficients = vector.reshape(2, 2)
    determinant_value = 2 * abs(np.linalg.det(coefficients))
    overlap_value = abs(np.vdot(vector, flip @ vector.conj()))
    projector = np.outer(vector, vector.conj())
    assert np.isclose(determinant_value, overlap_value, atol=1e-12)
    assert np.isclose(concurrence(projector), determinant_value, atol=1e-7)
    reduced = coefficients @ coefficients.conj().T
    eigenvalues = np.linalg.eigvalsh(reduced)
    direct_entropy = -sum(q * np.log2(q) for q in eigenvalues if q > 0)
    argument = (1 + np.sqrt(max(0, 1 - determinant_value**2))) / 2
    assert np.isclose(direct_entropy, binary_entropy(argument), atol=1e-12)
print("Twenty-four seeded complex pure-state checks passed: True")
