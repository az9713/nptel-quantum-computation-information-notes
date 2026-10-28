"""Toy RSA and exhaustive small order-to-factor classifications."""

from collections import Counter
from math import gcd


def order(base, modulus):
    """Return the exact order of a unit, using a small exhaustive loop."""
    if gcd(base, modulus) != 1:
        raise ValueError("Multiplicative order requires a unit")
    value = 1
    for exponent in range(1, modulus + 1):
        value = value*base % modulus
        if value == 1:
            return exponent
    raise AssertionError("Finite unit-group order was not found")


def classify(base, modulus):
    divisor = gcd(base, modulus)
    if divisor != 1:
        return "direct-gcd", divisor, modulus//divisor
    period = order(base, modulus)
    if period % 2:
        return "odd-order", period, None
    half_power = pow(base, period//2, modulus)
    if half_power == modulus - 1:
        return "minus-one", period, None
    left = gcd(half_power - 1, modulus)
    right = gcd(half_power + 1, modulus)
    assert 1 < left < modulus and 1 < right < modulus
    return "useful-order", period, tuple(sorted((left, right)))


def main():
    p, q, public_exponent, message = 3, 7, 5, 4
    modulus = p*q
    totient = (p - 1)*(q - 1)
    private_exponent = pow(public_exponent, -1, totient)
    cipher = pow(message, public_exponent, modulus)
    recovered = pow(cipher, private_exponent, modulus)
    assert recovered == message
    assert all(pow(pow(m, public_exponent, modulus),
                   private_exponent, modulus) == m
               for m in range(modulus))
    print(f"Toy RSA: N={modulus}, phi={totient}, "
          f"e={public_exponent}, d={private_exponent}")
    print(f"Message {message} -> ciphertext {cipher} -> {recovered}")
    print(f"All {modulus} message representatives decrypt correctly")
    for base in [2, 7]:
        values = [pow(base, x, 15) for x in range(8)]
        print(f"N=15, a={base}: powers={values}, order={order(base, 15)}")
    for modulus in [15, 21, 33]:
        counts = Counter(classify(base, modulus)[0]
                         for base in range(2, modulus))
        print(f"N={modulus}: " + ", ".join(
            f"{key}={counts[key]}" for key in sorted(counts)))
    for base in [2, 4, 14]:
        print(f"N=15, a={base}: {classify(base, 15)}")
    print(f"N=21, a=4: {classify(4, 21)}")
    assert order(4, 21) == 3
    print("Binary input lengths: " + ", ".join(
        f"{n}:{n.bit_length()}" for n in [15, 21, 33, 255, 256]))


if __name__ == "__main__":
    main()
