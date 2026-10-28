"""Exact toy order-finding probabilities and checked rational reconstruction.

NumPy is the only nonstandard dependency. This is a dense state simulation,
not a scalable implementation of modular-exponentiation gates.
"""

from math import gcd

import numpy as np


def continued_fraction(numerator, denominator):
    terms, fractions = [], []
    p_old, p_now, q_old, q_now = 0, 1, 1, 0
    while denominator:
        coefficient, remainder = divmod(numerator, denominator)
        terms.append(coefficient)
        p_old, p_now = p_now, coefficient * p_now + p_old
        q_old, q_now = q_now, coefficient * q_now + q_old
        fractions.append((p_now, q_now))
        numerator, denominator = denominator, remainder
    return terms, fractions


def exact_order(a, modulus):
    assert gcd(a, modulus) == 1
    value = 1
    for exponent in range(1, modulus):
        value = value * a % modulus
        if value == 1:
            return exponent
    raise AssertionError("A unit must return to one")


def factor_from_return(a, modulus, candidate):
    """A return exponent need not be the minimal order; check every branch."""
    if candidate % 2 or pow(a, candidate, modulus) != 1:
        return None
    half = pow(a, candidate // 2, modulus)
    if half in (1, modulus - 1):
        return None
    factors = (gcd(half - 1, modulus), gcd(half + 1, modulus))
    if all(1 < factor < modulus for factor in factors):
        return tuple(sorted(factors))
    return None


def reconstruct(y, size, a, modulus, max_multiple=1):
    if y == 0:
        return None
    _, fractions = continued_fraction(y, size)
    for _, denominator in fractions:
        if not 1 < denominator < modulus:
            continue
        for multiplier in range(1, max_multiple + 1):
            candidate = denominator * multiplier
            if candidate >= modulus:
                continue
            factors = factor_from_return(a, modulus, candidate)
            if factors is not None:
                return candidate, factors
    return None


def distribution(a, modulus, size):
    target_size = 1 << (modulus - 1).bit_length()
    joint = np.zeros((size, target_size), dtype=complex)
    for x in range(size):
        joint[x, pow(a, x, modulus)] = 1 / np.sqrt(size)
    labels = np.arange(size)
    fourier = np.exp(2j * np.pi * np.outer(labels, labels) / size)
    fourier /= np.sqrt(size)
    transformed = fourier @ joint
    marginal = np.sum(np.abs(transformed) ** 2, axis=1)

    # Independently sum conditional coset distributions with Born weights.
    mixture = np.zeros(size)
    for z in range(target_size):
        column = joint[:, z]
        weight = np.vdot(column, column).real
        if weight:
            conditional = column / np.sqrt(weight)
            mixture += weight * np.abs(fourier @ conditional) ** 2
    assert np.allclose(mixture, marginal, atol=1e-12)
    assert np.isclose(marginal.sum(), 1)
    return marginal


def main():
    assert continued_fraction(15, 4) == ([3, 1, 3], [(3, 1), (4, 1), (15, 4)])
    terms, fractions = continued_fraction(127, 256)
    print(f"127/256 continued fraction: {terms}")
    print(f"127/256 convergents: {fractions}")
    for modulus, a, size in ((15, 7, 256), (21, 2, 512)):
        order = exact_order(a, modulus)
        probabilities = distribution(a, modulus, size)
        largest = np.argsort(-probabilities, kind="stable")[:order]
        print(f"N={modulus}, a={a}, M={size}: reference order={order}")
        print("Largest bins: " + ", ".join(
            f"{y}:{probabilities[y]:.6f}" for y in sorted(largest)))
        print("Discarded B and unrecorded B measurement give equal marginals")
        for budget in (1, 2):
            success_mass = sum(
                probability for y, probability in enumerate(probabilities)
                if reconstruct(y, size, a, modulus, budget) is not None
            )
            print(f"Multiple budget={budget}: "
                  f"factor-success mass={success_mass:.6f}")
        if modulus == 15:
            expected = np.zeros(size)
            expected[[0, 64, 128, 192]] = 0.25
            assert np.allclose(probabilities, expected, atol=1e-12)
            print(f"P(127)={probabilities[127]:.6f}; "
                  f"P(128)={probabilities[128]:.6f}")
            assert reconstruct(128, size, a, modulus, 1) is None
            assert reconstruct(128, size, a, modulus, 2) == (4, (3, 5))
    assert pow(7, 8, 15) == 1
    assert factor_from_return(7, 15, 8) is None
    assert factor_from_return(7, 15, 4) == (3, 5)
    print("Return exponent 8 rejected: half-power is +1 modulo 15")
    print("Return exponent 4 accepted: verified factors are 3 and 5")


if __name__ == "__main__":
    main()
