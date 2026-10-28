"""Dimensionless entropy and reversible ideal-gas work calculations."""
import numpy as np


def entropy(p):
    probabilities = np.array([p, 1 - p], dtype=float)
    positive = probabilities[probabilities > 0]
    return -np.sum(positive * np.log(positive))


for p in [0.0, 0.1, 0.25, 0.5, 1.0]:
    value = entropy(p)
    assert value >= -1e-14 and value <= np.log(2) + 1e-14
    assert np.isclose(value, entropy(1 - p))
    print(f"P(bit=1)={p:.2f}: reset bound Q/(k_B*T)>={value:.6f}")
volume = np.linspace(1.0, 0.5, 10001)
work = -np.trapezoid(1 / volume, volume)
assert np.isclose(work, np.log(2), atol=1e-8)
print(f"one-particle reversible compression W/(k_B*T)={work:.6f}")
epsilon = 0.01
imperfect = np.log(2) - entropy(epsilon)
assert imperfect < np.log(2)
print(f"1-percent residual error: entropy bound={imperfect:.6f}")
q_bath = np.log(2) + 0.2
memory_entropy_change = -np.log(2)
total = q_bath + memory_entropy_change
assert np.isclose(total, 0.2)
print(f"example total entropy production / k_B={total:.6f}")
print("all dimensionless entropy and work checks passed")
