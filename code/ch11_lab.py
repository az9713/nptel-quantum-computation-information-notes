"""Ideal polarization Bell analysis and explicit conditioning budgets."""

from math import sqrt

import numpy as np


def two_photon_output(terms, mixing):
    """Map creation-operator pairs to normalized output Fock amplitudes."""
    amplitudes = {}
    for coefficient, first, second in terms:
        for i in range(4):
            for j in range(4):
                occupation = tuple(int(k == i) + int(k == j)
                                   for k in range(4))
                factor = sqrt(2) if i == j else 1
                value = coefficient*mixing[i, first]*mixing[j, second]
                amplitudes[occupation] = (
                    amplitudes.get(occupation, 0) + factor*value)
    return {key: value for key, value in amplitudes.items()
            if abs(value) > 1e-12}


def main():
    basis = np.eye(4)
    bells = [
        (basis[:, 0] + basis[:, 3])/sqrt(2),
        (basis[:, 0] - basis[:, 3])/sqrt(2),
        (basis[:, 1] + basis[:, 2])/sqrt(2),
        (basis[:, 1] - basis[:, 2])/sqrt(2),
    ]
    projectors = [np.outer(v, v.conj()) for v in bells]
    assert np.allclose(sum(projectors), np.eye(4))
    for i, first in enumerate(projectors):
        for j, second in enumerate(projectors):
            assert np.allclose(first @ second, first if i == j else 0)
    print("Bell projectors: orthogonal, complete, trace one")

    # Input modes aH,aV,bH,bV; output modes cH,cV,dH,dV.
    mixing = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                       [1, 0, -1, 0], [0, 1, 0, -1]])/sqrt(2)
    assert np.allclose(mixing.conj().T @ mixing, np.eye(4))
    root = 1/sqrt(2)
    inputs = {
        "Phi+": [(root, 0, 2), (root, 1, 3)],
        "Phi-": [(root, 0, 2), (-root, 1, 3)],
        "Psi+": [(root, 0, 3), (root, 1, 2)],
        "Psi-": [(root, 0, 3), (-root, 1, 2)],
    }
    distributions = {}
    for name, terms in inputs.items():
        output = two_photon_output(terms, mixing)
        probabilities = {key: abs(value)**2
                         for key, value in output.items()}
        assert np.isclose(sum(probabilities.values()), 1)
        coincidence = sum(value for key, value in probabilities.items()
                          if sum(key[:2]) == sum(key[2:]) == 1)
        distributions[name] = probabilities
        print(f"{name}: patterns={len(output)}, "
              f"different-port probability={coincidence:.6f}")
    assert distributions["Phi+"] == distributions["Phi-"]
    identified = 0
    for name, distribution in distributions.items():
        for pattern, probability in distribution.items():
            compatible = sum(pattern in other
                             for other in distributions.values())
            if compatible == 1:
                identified += probability/4
    print(f"Equiprobable Bell-input success: {identified:.6f}")

    for overlap in [0, 0.5, 1]:
        rho = np.zeros((4, 4))
        rho[1, 1] = rho[2, 2] = 0.5
        rho[1, 2] = rho[2, 1] = overlap/2
        fidelity = np.trace(rho @ projectors[2]).real
        assert np.linalg.eigvalsh(rho).min() >= -1e-12
        print(f"Environment overlap={overlap:.1f}: "
              f"Psi+ fidelity={fidelity:.6f}")

    pair_probability, efficiency, trials = 0.02, 0.75, 10000
    accepted = pair_probability*identified*efficiency**2
    print(f"Illustrative accepted-event probability: {accepted:.6f}")
    print(f"Expected accepted events in {trials} trials: "
          f"{trials*accepted:.2f}")


if __name__ == "__main__":
    main()
