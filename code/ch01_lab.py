"""Finite checks for qubit bases, measurement, and storage accounting."""
import numpy as np

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)
for n in (1, 2, 3):
    vectors = []
    for x in range(2**n):
        vector = np.array([1], dtype=complex)
        for bit in f"{x:0{n}b}":
            vector = np.kron(vector, zero if bit == "0" else one)
        vectors.append(vector)
    basis = np.column_stack(vectors)
    assert np.allclose(basis.conj().T @ basis, np.eye(2**n))
    assert np.allclose(basis, np.eye(2**n))
    print(f"n={n}: {len(vectors)} orthonormal basis vectors")

left = np.array([np.sqrt(3) / 2, 0.5j])
right = np.array([1, -1]) / np.sqrt(2)
psi = np.kron(left, right)
probabilities = np.abs(psi)**2
assert np.isclose(np.vdot(psi, psi), 1)
assert np.allclose(probabilities, [0.375, 0.375, 0.125, 0.125])
print("product probabilities:", np.round(probabilities, 6))
print("first-qubit marginal:", probabilities.reshape(2, 2).sum(axis=1))

hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
plus = np.array([1, 1]) / np.sqrt(2)
minus = np.array([1, -1]) / np.sqrt(2)
assert np.allclose(np.abs(plus)**2, np.abs(minus)**2)
assert np.allclose(np.abs(hadamard @ plus)**2, [1, 0])
assert np.allclose(np.abs(hadamard @ minus)**2, [0, 1])
print("Z probabilities, plus and minus: [0.5 0.5], [0.5 0.5]")
print("after Hadamard, plus and minus: [1. 0.], [0. 1.]")
for n in (30, 40, 50):
    storage = 16 * 2**n
    print(f"n={n}: dense complex128 state requires {storage} bytes")
shots = 1024
p = probabilities[0]
standard_deviation = np.sqrt(p * (1 - p) / shots)
print(f"1024-shot frequency standard deviation: {standard_deviation:.6f}")
assert (1023).bit_length() == 10 and (1024).bit_length() == 11
print("all finite checks passed")
