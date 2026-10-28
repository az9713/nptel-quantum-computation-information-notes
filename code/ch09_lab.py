"""Deterministic local bounds, quantum correlators, and angle conventions."""
import itertools
import numpy as np

x = np.array([[0, 1], [1, 0]], dtype=complex)
y = np.array([[0, -1j], [1j, 0]], dtype=complex)
z = np.diag([1, -1]).astype(complex)
identity = np.eye(2)
phi = np.array([1, 0, 0, 1], dtype=complex)/np.sqrt(2)
singlet = np.array([0, 1, -1, 0], dtype=complex)/np.sqrt(2)
local = [a0*(b0 + b1) + a1*(b0 - b1)
         for a0, a1, b0, b1 in itertools.product([-1, 1], repeat=4)]
assert set(local) == {-2, 2}
print("local CHSH values:", sorted(set(local)))


def expectation(state, operator):
    return float(np.vdot(state, operator @ state).real)


a0, a1 = z, x
b0, b1 = (z + x)/np.sqrt(2), (z - x)/np.sqrt(2)
terms = [expectation(phi, np.kron(a, b))
         for a, b in [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]]
value = terms[0] + terms[1] + terms[2] - terms[3]
assert np.isclose(value, 2*np.sqrt(2))
print("quantum correlators:", " ".join(f"{v:.6f}" for v in terms))
print(f"quantum CHSH={value:.6f}")
for a in [a0, a1]:
    assert abs(expectation(phi, np.kron(a, identity))) < 1e-12
for b in [b0, b1]:
    assert abs(expectation(phi, np.kron(identity, b))) < 1e-12
print("all chosen single-party expectations=0")


def spin(angle):
    return np.sin(angle)*x + np.cos(angle)*z


for angle in [0, np.pi/3, np.pi/2, np.pi]:
    direct = expectation(singlet, np.kron(z, spin(angle)))
    assert np.isclose(direct, -np.cos(angle))
    print(f"singlet angle={np.degrees(angle):.0f} deg: E={direct:.6f}")

left = (1 - np.cos(2*np.pi/3))/4
right = 2*(1 - np.cos(np.pi/3))/4
print(f"three-setting test: left={left:.6f}, right={right:.6f}, "
      f"violation={left - right:.6f}")

grid = np.linspace(0, np.pi/2, 1001)
values = 2*(np.cos(grid) + np.sin(grid))
best = int(np.argmax(values))
print(f"CHSH angle optimum={np.degrees(grid[best]):.6f} deg, "
      f"value={values[best]:.6f}")

cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
plus = np.array([1, 1], dtype=complex)/np.sqrt(2)
zero = np.array([1, 0], dtype=complex)
assert np.allclose(cnot.conj().T @ cnot, np.eye(4))
assert np.allclose(cnot @ np.kron(plus, zero), phi)
assert np.allclose(cnot @ np.kron(zero, plus), np.kron(zero, plus))
print("CNOT: |+0> becomes Bell; |0+> remains product")
