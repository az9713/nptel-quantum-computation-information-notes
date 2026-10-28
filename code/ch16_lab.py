"""Check trace, positivity, and Bloch formulas for qubit noise maps."""
import numpy as np

identity = np.eye(2)
x = np.array([[0, 1], [1, 0]], complex)
y = np.array([[0, -1j], [1j, 0]], complex)
z = np.diag([1, -1])
paulis = [x, y, z]
r = np.array([0.6, 0.0, 0.8])
rho = (identity + sum(v * p for v, p in zip(r, paulis))) / 2


def bloch(state):
    return np.array([np.trace(state @ p).real for p in paulis])


def apply(operators, state):
    assert np.allclose(sum(k.conj().T @ k for k in operators), identity)
    out = sum(k @ state @ k.conj().T for k in operators)
    assert np.isclose(np.trace(out), 1)
    assert np.linalg.eigvalsh(out).min() > -1e-12
    return out


for p in np.linspace(0, 1, 21):
    operators = [np.sqrt(1 - 3 * p / 4) * identity]
    operators += [np.sqrt(p / 4) * a for a in paulis]
    assert np.allclose(bloch(apply(operators, rho)), (1 - p) * r)
    operators = [np.diag([1, np.sqrt(1 - p)]),
                 np.array([[0, np.sqrt(p)], [0, 0]])]
    expected = [np.sqrt(1 - p) * r[0],
                np.sqrt(1 - p) * r[1], (1 - p) * r[2] + p]
    assert np.allclose(bloch(apply(operators, rho)), expected)
gamma = 0.36
operators = [np.diag([1, np.sqrt(1 - gamma)]),
             np.array([[0, np.sqrt(gamma)], [0, 0]])]
damped = apply(operators, rho)
mixed = apply(operators, identity / 2)
eta = 0.5
dephased = apply([np.sqrt((1 + eta) / 2) * identity,
                  np.sqrt((1 - eta) / 2) * z], rho)
print("Input Bloch vector:", r.tolist())
print("Amplitude damping:", bloch(damped).round(6).tolist())
print("Damped maximally mixed state:", bloch(mixed).round(6).tolist())
print("Pure dephasing:", bloch(dephased).round(6).tolist())
print("Twenty-one-point channel sweeps passed: True")
