"""Ensemble equivalence, state validity, and isolated density evolution."""
import numpy as np

identity = np.eye(2, dtype=complex)
x = np.array([[0, 1], [1, 0]], complex)
y = np.array([[0, -1j], [1j, 0]])
z = np.diag([1, -1])
paulis = [x, y, z]
zero, one = np.eye(2, dtype=complex)
plus, minus = (zero + one) / np.sqrt(2), (zero - one) / np.sqrt(2)


def projector(vector):
    return np.outer(vector, vector.conj())


def check_state(state):
    assert np.allclose(state, state.conj().T, atol=1e-12)
    assert np.isclose(np.trace(state), 1)
    assert np.linalg.eigvalsh(state).min() > -1e-12


def bloch(state):
    return np.array([np.trace(state @ p).real for p in paulis])


def purity(state):
    return np.trace(state @ state).real


z_ensemble = (projector(zero) + projector(one)) / 2
x_ensemble = (projector(plus) + projector(minus)) / 2
assert np.allclose(z_ensemble, x_ensemble)
for observable in paulis:
    assert np.isclose(np.trace(z_ensemble @ observable),
                      np.trace(x_ensemble @ observable))
pure_x_probability = np.trace(projector(plus) @ projector(plus)).real
mixed_x_probability = np.trace(z_ensemble @ projector(plus)).real
print("Hidden Z and X ensembles have identical density operators: True")
print(f"X-positive probabilities: pure={pure_x_probability:.6f}, "
      f"mixed={mixed_x_probability:.6f}")
biased = np.diag([0.6, 0.4])
print(f"Six-up/four-down purity = {purity(biased):.6f}")
psi = np.array([np.sqrt(3) / 2, 0.5j])
initial = 0.7 * projector(psi) + 0.3 * projector(zero)
check_state(initial)
omega = 1.3  # radians per second; H_phys/hbar has inverse-second units.
axis = np.array([0.6, 0.0, 0.8])
axis_matrix = sum(v * p for v, p in zip(axis, paulis))
hamiltonian_frequency = omega * axis_matrix / 2


def evolved(time):
    unitary = (np.cos(omega * time / 2) * identity
               - 1j * np.sin(omega * time / 2) * axis_matrix)
    return unitary @ initial @ unitary.conj().T


initial_spectrum = np.linalg.eigvalsh(initial)
initial_energy = np.trace(initial @ hamiltonian_frequency)
largest_residual = 0.0
step = 1e-5
for time in np.linspace(0, 4, 41):
    state = evolved(time)
    check_state(state)
    assert np.allclose(np.linalg.eigvalsh(state), initial_spectrum)
    assert np.isclose(purity(state), purity(initial))
    assert np.isclose(np.trace(state @ hamiltonian_frequency), initial_energy)
    finite_difference = ((evolved(time + step) - evolved(time - step))
                         / (2 * step))
    direct = -1j * (hamiltonian_frequency @ state
                    - state @ hamiltonian_frequency)
    residual = np.max(np.abs(finite_difference - direct))
    largest_residual = max(largest_residual, residual)
    assert residual < 1e-9
print("Initial complex-state Bloch vector:", bloch(initial).round(6).tolist())
print(f"Unitary-invariant purity = {purity(initial):.6f}")
print("Forty-one-point spectrum, trace, energy, and purity checks "
      "passed: True")
print("Central-difference commutator residual below 1e-9:",
      largest_residual < 1e-9)
