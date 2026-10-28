"""Compare a unitary dilation, its Kraus map, and ancillary positivity."""
import numpy as np


def partial_trace_environment(rho):
    return np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)


gamma = 0.36
c = np.sqrt(1 - gamma)
s = np.sqrt(gamma)
unitary = np.array([[1, 0, 0, 0], [0, c, s, 0],
                    [0, -s, c, 0], [0, 0, 0, 1]], complex)
kraus = [unitary.reshape(2, 2, 2, 2)[:, j, :, 0] for j in range(2)]
identity = np.eye(2)
assert np.allclose(unitary.conj().T @ unitary, np.eye(4))
assert np.allclose(sum(k.conj().T @ k for k in kraus), identity)
psi = np.array([1, 1j]) / np.sqrt(2)
rho = np.outer(psi, psi.conj())
joint = np.kron(rho, np.diag([1, 0]))
reduced = partial_trace_environment(unitary @ joint @ unitary.conj().T)
mapped = sum(k @ rho @ k.conj().T for k in kraus)
assert np.allclose(reduced, mapped)
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
bell_rho = np.outer(bell, bell)
enlarged = sum(np.kron(k, identity) @ bell_rho
               @ np.kron(k, identity).conj().T for k in kraus)
assert np.linalg.eigvalsh(enlarged).min() > -1e-12
transposed = bell_rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1)
pt_eigenvalues = np.linalg.eigvalsh(transposed.reshape(4, 4))
assert np.allclose(pt_eigenvalues, [-0.5, 0.5, 0.5, 0.5])
print("Dilation and Kraus outputs agree: True")
print("Output populations:", np.diag(mapped).real.round(6).tolist())
print("Output coherence magnitude:", f"{abs(mapped[0, 1]):.6f}")
print("Ancillary Kraus output is positive: True")
print("Partial-transpose eigenvalues:", pt_eigenvalues.round(6).tolist())
