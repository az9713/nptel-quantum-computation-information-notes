"""Schmidt reconstruction in real and genuinely complex product bases."""
import numpy as np


def analyze(name, coefficients):
    assert abs(np.linalg.norm(coefficients) - 1) < 1e-12
    dim_a, dim_b = coefficients.shape
    left, singular, right_h = np.linalg.svd(coefficients, full_matrices=False)
    reconstructed = sum(singular[j]*np.outer(left[:, j], right_h[j, :])
                        for j in range(len(singular)))
    assert np.allclose(reconstructed, coefficients)
    rho_a = coefficients @ coefficients.conj().T
    rho_b = coefficients.T @ coefficients.conj()
    spectrum_a = np.linalg.eigvalsh(rho_a)[::-1]
    spectrum_b = np.linalg.eigvalsh(rho_b)[::-1]
    assert np.allclose(spectrum_a[:len(singular)], singular**2)
    assert np.allclose(spectrum_b[:len(singular)], singular**2)
    assert np.allclose(spectrum_b[len(singular):], 0)
    positive = singular[singular > 1e-12]**2
    entropy = float(-np.sum(positive*np.log2(positive)))
    rank = int(np.sum(singular > 1e-12))
    print(f"{name}: dimensions={dim_a}x{dim_b}, Schmidt rank={rank}")
    print("  singular values:", np.array2string(singular, precision=6))
    print("  rho_A spectrum:", np.array2string(spectrum_a, precision=6))
    print("  rho_B spectrum:", np.array2string(spectrum_b, precision=6))
    print(f"  entanglement entropy={max(0.0, entropy):.6f} bits")
    wrong = sum(singular[j]*np.outer(left[:, j], right_h[j, :].conj())
                for j in range(len(singular)))
    print("  wrong-conjugation residual="
          f"{np.linalg.norm(wrong - coefficients):.6f}")
    return singular


real = np.array([[1, 1], [1, 0]], dtype=complex)/np.sqrt(3)
analyze("three-term state", real)
complex_state = np.array([[1, 1j, 0], [0, 1, 1j]], dtype=complex)/2
analyze("complex qubit-qutrit", complex_state)
product = np.array([[1, 1], [1, 1]], dtype=complex)/2
analyze("product state", product)
bell = np.eye(2, dtype=complex)/np.sqrt(2)
analyze("Bell state", bell)
expected = np.array([(3 + np.sqrt(5))/6, (3 - np.sqrt(5))/6])
assert np.allclose(np.linalg.eigvalsh(real @ real.conj().T)[::-1], expected)
print("three-term reduced eigenvalues:",
      np.array2string(expected, precision=6))
