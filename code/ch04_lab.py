"""Spinor rotations, Bloch vectors, and nonsingular/singular Euler recovery."""
import numpy as np

identity = np.eye(2, dtype=complex)
x = np.array([[0, 1], [1, 0]], complex)
y = np.array([[0, -1j], [1j, 0]])
z = np.diag([1, -1])
paulis = [x, y, z]


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    assert np.isclose(axis @ axis, 1)
    generator = sum(v * p for v, p in zip(axis, paulis))
    return np.cos(angle / 2) * identity - 1j * np.sin(angle / 2) * generator


def bloch(vector):
    return np.array([np.vdot(vector, p @ vector).real for p in paulis])


def euler(unitary):
    assert np.allclose(unitary.conj().T @ unitary, identity)
    phase = np.angle(np.linalg.det(unitary)) / 2
    special = np.exp(-1j * phase) * unitary
    a, b = special[:, 0]
    middle = 2 * np.arctan2(abs(b), abs(a))
    if abs(b) < 1e-12:
        left, right = -2 * np.angle(a), 0.0
    elif abs(a) < 1e-12:
        left, right = 2 * np.angle(b), 0.0
    else:
        left = np.angle(b) - np.angle(a)
        right = -np.angle(b) - np.angle(a)
    return phase, left, middle, right


def reconstruct(angles):
    phase, left, middle, right = angles
    return (np.exp(1j * phase) * rotation([0, 0, 1], left)
            @ rotation([0, 1, 0], middle) @ rotation([0, 0, 1], right))


theta, phi = np.pi / 3, np.pi / 2
state = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
assert np.allclose(bloch(state), [0, np.sqrt(3) / 2, 0.5])
print("Complex-state Bloch vector:", bloch(state).round(6).tolist())
random = np.random.default_rng(20261028)
largest = 0.0
for _ in range(96):
    axis = random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = random.uniform(-4 * np.pi, 4 * np.pi)
    vector = random.normal(size=2) + 1j * random.normal(size=2)
    vector /= np.linalg.norm(vector)
    gate = rotation(axis, angle)
    r = bloch(vector)
    expected = (np.cos(angle) * r
                + (1 - np.cos(angle)) * (axis @ r) * axis
                + np.sin(angle) * np.cross(axis, r))
    assert np.allclose(bloch(gate @ vector), expected, atol=1e-12)
    phase = random.uniform(-np.pi, np.pi)
    gate *= np.exp(1j * phase)
    residual = np.max(np.abs(gate - reconstruct(euler(gate))))
    largest = max(largest, residual)
    assert residual < 1e-12
for middle in [0.0, np.pi]:
    gate = reconstruct([0.2, 0.7, middle, -0.3])
    assert np.allclose(gate, reconstruct(euler(gate)), atol=1e-12)
hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
quarter = rotation([0, 1, 0], np.pi / 2)
assert np.allclose(1j * rotation([1, 0, 0], np.pi) @ quarter, hadamard)
assert not np.allclose(quarter, hadamard)
assert np.allclose(rotation([0, 0, 1], 2 * np.pi), -identity)
assert np.allclose(bloch(-state), bloch(state))
print("Ninety-six complex rotation/Euler checks passed: True")
print("Euler endpoint branches passed: True")
print("Hadamard equals i Rx(pi) Ry(pi/2): True")
print("A 2pi spinor rotation is -I; its Bloch action is identity: True")
print("Largest reconstruction residual below 1e-12:", largest < 1e-12)
