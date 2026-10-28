"""Deutsch query calculation, including a controlled preparation error."""
import numpy as np

h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
identity = np.eye(2)
initial = np.array([0, 1, 0, 0], dtype=complex)
minus = np.array([1, -1]) / np.sqrt(2)
plus = np.array([1, 1]) / np.sqrt(2)


def oracle(truth):
    matrix = np.zeros((4, 4))
    for x in (0, 1):
        for y in (0, 1):
            matrix[2 * x + (y ^ truth[x]), 2 * x + y] = 1
    return matrix


for truth in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    uf = oracle(truth)
    assert np.allclose(uf.T @ uf, np.eye(4))
    assert np.allclose(uf @ uf, np.eye(4))
    final = np.kron(h, identity) @ uf @ np.kron(h, h) @ initial
    marginal = np.abs(final.reshape(2, 2))**2
    marginal = marginal.sum(axis=1)
    parity = truth[0] ^ truth[1]
    assert np.isclose(marginal[parity], 1)
    print(f"f={truth}: P(first=0)={marginal[0]:.6f}, "
          f"P(first=1)={marginal[1]:.6f}")

theta = 0.3
ancilla = np.cos(theta) * minus + np.sin(theta) * plus
state = np.kron(plus, ancilla)
final = np.kron(h, identity) @ oracle((0, 1)) @ state
success = (np.abs(final.reshape(2, 2))**2).sum(axis=1)[1]
assert np.isclose(success, np.cos(theta)**2)
print(f"balanced success at preparation angle 0.3: {success:.6f}")
blind = np.kron(h, identity) @ oracle((0, 1)) @ np.kron(plus, plus)
assert np.isclose((np.abs(blind.reshape(2, 2))**2).sum(axis=1)[0], 1)
print("plus-target balanced input incorrectly returns 0 with probability 1")
print("all finite oracle and probability checks passed")
