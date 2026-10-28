"""Positive-sign QFT: dense matrix, gate circuit, and finite periods."""

import numpy as np


def fourier(size):
    labels = np.arange(size)
    return np.exp(2j*np.pi*np.outer(labels, labels)/size)/np.sqrt(size)


def circuit(state, qubits, cutoff=None):
    state = np.array(state, dtype=complex, copy=True)
    size = 1 << qubits
    assert state.shape == (size,)
    labels = np.arange(size)
    for target in range(qubits):
        target_bit = 1 << (qubits - 1 - target)
        for index in range(size):
            if not index & target_bit:
                other = index | target_bit
                low, high = state[index], state[other]
                state[index] = (low + high)/np.sqrt(2)
                state[other] = (low - high)/np.sqrt(2)
        for control in range(target + 1, qubits):
            rotation_index = control - target + 1
            if cutoff is not None and rotation_index > cutoff:
                continue
            control_bit = 1 << (qubits - 1 - control)
            selected = ((labels & target_bit) != 0)
            selected &= ((labels & control_bit) != 0)
            angle = 2*np.pi/(1 << rotation_index)
            state[selected] *= np.exp(1j*angle)
    reversed_labels = np.array([
        int(f"{i:0{qubits}b}"[::-1], 2) for i in range(size)])
    return state[reversed_labels]


def progression(size, period, offset):
    state = np.zeros(size, dtype=complex)
    labels = np.arange(offset, size, period)
    state[labels] = 1/np.sqrt(len(labels))
    return state, len(labels)


def main():
    for qubits in range(1, 6):
        size = 1 << qubits
        dense = fourier(size)
        basis = np.eye(size)
        gates = np.column_stack([
            circuit(basis[:, i], qubits) for i in range(size)])
        assert np.allclose(dense.conj().T @ dense, basis, atol=1e-12)
        assert np.allclose(gates, dense, atol=1e-12)
        phases = qubits*(qubits - 1)//2
        print(f"n={qubits}, M={size}: circuit=dense, "
              f"H={qubits}, controlled phases={phases}, "
              f"swaps={qubits//2}")
    factors = [np.array([1, np.exp(2j*np.pi*5/(1 << k))])
               / np.sqrt(2) for k in [1, 2, 3]]
    product = np.kron(np.kron(factors[0], factors[1]), factors[2])
    assert np.allclose(product, fourier(8)[:, 5])
    print("x=5, M=8: tensor-product phase reconstruction passed")
    for period in [4, 3]:
        state, length = progression(16, period, 1)
        probabilities = np.abs(fourier(16) @ state)**2
        assert np.isclose(probabilities.sum(), 1)
        peaks = sorted(range(16), key=lambda i: (-probabilities[i], i))
        peaks = sorted(peaks[:period])
        print(f"M=16, period={period}, offset=1, length={length}")
        print("Largest bins: " + ", ".join(
            f"{i}:{probabilities[i]:.6f}" for i in peaks))
        if period == 4:
            expected = np.zeros(16)
            expected[::4] = 0.25
            assert np.allclose(probabilities, expected)
    qubits, cutoff = 5, 3
    basis = np.eye(1 << qubits)
    truncated = np.column_stack([
        circuit(basis[:, i], qubits, cutoff) for i in range(1 << qubits)])
    error = np.linalg.norm(truncated - fourier(1 << qubits), ord=2)
    bound = 2*np.pi*qubits/(1 << cutoff)
    assert error <= bound + 1e-12
    print(f"Truncated n={qubits}, cutoff={cutoff}: "
          f"operator error={error:.6f}, bound={bound:.6f}")


if __name__ == "__main__":
    main()
