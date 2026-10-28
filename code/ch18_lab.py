"""Construct a trine POVM and two instruments with identical effects."""
import numpy as np

identity = np.eye(2)
x = np.array([[0, 1], [1, 0]])
vectors = [np.array([1, np.exp(2j * np.pi * j / 3)]) / np.sqrt(2)
           for j in range(3)]
projectors = [np.outer(v, v.conj()) for v in vectors]
effects = [2 * p / 3 for p in projectors]
operators = [np.sqrt(2 / 3) * p for p in projectors]
assert np.allclose(sum(effects), identity)
assert all(np.linalg.eigvalsh(e).min() > -1e-12 for e in effects)
rho = np.diag([1, 0])
probabilities = [np.trace(rho @ e).real for e in effects]
assert np.allclose(probabilities, np.ones(3) / 3)
altered = [x @ a for a in operators]
for a, b in zip(operators, altered):
    assert np.allclose(a.conj().T @ a, b.conj().T @ b)
j = 1
out_a = operators[j] @ rho @ operators[j].conj().T / probabilities[j]
out_b = altered[j] @ rho @ altered[j].conj().T / probabilities[j]
assert not np.allclose(out_a, out_b)
print("POVM probabilities:", np.round(probabilities, 6).tolist())
print("Same effects for both instruments: True")
print("Conditional coherence, first instrument:",
      f"{out_a[0, 1].real:.6f}{out_a[0, 1].imag:+.6f}j")
print("Conditional coherence, second instrument:",
      f"{out_b[0, 1].real:.6f}{out_b[0, 1].imag:+.6f}j")
psi_usd = np.array([1, 0], complex)
phi_usd = np.array([1, 1], complex) / np.sqrt(2)
overlap = abs(np.vdot(psi_usd, phi_usd))
scale = 1 / (1 + overlap)
f_zero = scale * (np.eye(2) - np.outer(psi_usd, psi_usd.conj()))
f_one = scale * (np.eye(2) - np.outer(phi_usd, phi_usd.conj()))
f_question = np.eye(2) - f_zero - f_one
for effect in [f_zero, f_one, f_question]:
    assert np.linalg.eigvalsh(effect).min() > -1e-12
assert np.allclose(f_zero @ psi_usd, 0)
assert np.allclose(f_one @ phi_usd, 0)
success = np.vdot(psi_usd, f_one @ psi_usd).real
assert np.isclose(success, 1 - overlap)
print(f"Unambiguous conclusive probability = {success:.6f}")
