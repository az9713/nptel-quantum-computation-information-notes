"""Entropy, reductions, and local measurement checks for two-qubit states."""
import numpy as np


def projector(vector):
    return np.outer(vector, vector.conj())


def reductions(rho, dim_a=2, dim_b=2):
    tensor = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    return np.einsum("abcb->ac", tensor), np.einsum("abad->bd", tensor)


def entropy(rho):
    assert np.allclose(rho, rho.conj().T)
    eigenvalues = np.linalg.eigvalsh(rho)
    assert eigenvalues.min() >= -1e-12
    assert abs(eigenvalues.sum() - 1) < 1e-12
    positive = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(positive*np.log2(positive)))


zero, one = np.eye(2, dtype=complex)
plus = (zero + one)/np.sqrt(2)
minus = (zero - one)/np.sqrt(2)
bell = (np.kron(zero, zero) + np.kron(one, one))/np.sqrt(2)
product = np.kron(plus, plus)
correlated = (projector(np.kron(zero, zero))
              + projector(np.kron(one, one)))/2
for name, rho in [("Bell", projector(bell)),
                  ("product", projector(product)),
                  ("classical mixture", correlated)]:
    rho_a, rho_b = reductions(rho)
    print(f"{name}: S(AB)={entropy(rho):.6f}, "
          f"S(A)={entropy(rho_a):.6f}, S(B)={entropy(rho_b):.6f}")
    for i in range(2):
        for j in range(2):
            observable = np.outer(np.eye(2)[i], np.eye(2)[j])
            assert np.allclose(np.trace(rho_a @ observable),
                               np.trace(rho @ np.kron(observable, np.eye(2))))

rho = projector(bell)
for name, basis in [("Z", [zero, one]), ("X", [plus, minus])]:
    averaged = np.zeros((2, 2), dtype=complex)
    for vector in basis:
        local = np.kron(np.eye(2), projector(vector))
        unnormalized = local @ rho @ local.conj().T
        reduced, _ = reductions(unnormalized)
        averaged += reduced
    assert np.allclose(averaged, np.eye(2)/2)
    print(f"unread {name}-measurement on B: A remains I/2")

print(f"S(diag(3/4,1/4))={entropy(np.diag([0.75, 0.25])):.6f}")
for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
    state = np.sqrt(value)*np.kron(zero, zero)
    state += np.sqrt(1 - value)*np.kron(one, one)
    rho_a, _ = reductions(projector(state))
    print(f"p={value:.2f}: local entropy={entropy(rho_a):.6f}")
