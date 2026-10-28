"""Compare a Lindblad matrix exponential with analytical decay laws."""
import numpy as np
from scipy.linalg import expm

identity = np.eye(2)
z = np.diag([1, -1])
lower = np.array([[0, 1], [0, 0]], complex)
gamma_one = 2.0
gamma_phi = 0.5
jumps = [np.sqrt(gamma_one) * lower, np.sqrt(gamma_phi / 2) * z]
generator = np.zeros((4, 4), complex)
for jump in jumps:
    norm = jump.conj().T @ jump
    generator += np.kron(jump.conj(), jump)
    generator -= (np.kron(identity, norm) + np.kron(norm.T, identity)) / 2
for row in range(2):
    for column in range(2):
        basis = np.zeros((2, 2), complex)
        basis[row, column] = 1
        direct = np.zeros((2, 2), complex)
        for jump in jumps:
            norm = jump.conj().T @ jump
            direct += jump @ basis @ jump.conj().T
            direct -= (norm @ basis + basis @ norm) / 2
        assert np.allclose(generator @ basis.ravel(order="F"),
                           direct.ravel(order="F"))
rho_initial = np.ones((2, 2)) / 2
for t in np.linspace(0, 3, 31):
    state = (expm(t * generator) @ rho_initial.ravel(order="F"))
    state = state.reshape(2, 2, order="F")
    population = 0.5 * np.exp(-gamma_one * t)
    coherence = 0.5 * np.exp(-(gamma_one / 2 + gamma_phi) * t)
    expected = np.array([[1 - population, coherence],
                         [coherence, population]])
    assert np.allclose(state, expected, atol=1e-12)
    assert np.isclose(np.trace(state), 1)
    assert np.linalg.eigvalsh(state).min() > -1e-12
t_one = 1 / gamma_one
t_two = 1 / (gamma_one / 2 + gamma_phi)
print(f"T1 = {t_one:.6f} s; T2 = {t_two:.6f} s")
print(f"At t=1 s: excited population = {0.5*np.exp(-2):.6f}")
print(f"At t=1 s: coherence magnitude = {0.5*np.exp(-1.5):.6f}")
print("Thirty-one-point exponential comparison passed: True")

# A second generator tests the finite-temperature solution including its phase.
gamma_down, gamma_up, omega = 3.0, 1.0, 0.7
thermal_jumps = [np.sqrt(gamma_down) * lower,
                 np.sqrt(gamma_up) * lower.conj().T,
                 np.sqrt(gamma_phi / 2) * z]
hamiltonian = -omega * z / 2  # H_phys/hbar, in inverse seconds.
thermal_generator = -1j * (np.kron(identity, hamiltonian)
                           - np.kron(hamiltonian.T, identity))
for jump in thermal_jumps:
    norm = jump.conj().T @ jump
    thermal_generator += np.kron(jump.conj(), jump)
    thermal_generator -= (np.kron(identity, norm)
                          + np.kron(norm.T, identity)) / 2
rate = gamma_down + gamma_up
equilibrium = gamma_up / rate
for t in np.linspace(0, 3, 31):
    state = expm(t * thermal_generator) @ rho_initial.ravel(order="F")
    state = state.reshape(2, 2, order="F")
    pop = equilibrium + (0.5 - equilibrium) * np.exp(-rate * t)
    coh = 0.5 * np.exp((1j * omega - rate / 2 - gamma_phi) * t)
    expected = np.array([[1 - pop, coh], [coh.conjugate(), pop]])
    assert np.allclose(state, expected, atol=1e-12)
    assert np.isclose(np.trace(state), 1)
    assert np.linalg.eigvalsh(state).min() > -1e-12
print(f"Thermal equilibrium excited population = {equilibrium:.6f}")
print("Thermal population, phase, and coherence grid passed: True")
