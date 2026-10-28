"""Exact small prefix-code checks and a finite-block compression comparison."""
import heapq
import itertools
import math


def entropy(probabilities):
    return -sum(p*math.log2(p) for p in probabilities if p > 0)


def huffman_lengths(probabilities):
    heap = [(p, i, {i: 0}) for i, p in enumerate(probabilities) if p > 0]
    heapq.heapify(heap)
    serial = 1 + max((item[1] for item in heap), default=-1)
    if len(heap) == 1:
        return {heap[0][1]: 0}
    while len(heap) > 1:
        p, _, left = heapq.heappop(heap)
        q, _, right = heapq.heappop(heap)
        depths = {i: length + 1 for i, length in left.items()}
        depths.update({i: length + 1 for i, length in right.items()})
        heapq.heappush(heap, (p + q, serial, depths))
        serial += 1
    return heap[0][2]


def inspect(probabilities):
    lengths = huffman_lengths(probabilities)
    mean = sum(probabilities[i]*length for i, length in lengths.items())
    kraft = sum(2.0**(-length) for length in lengths.values())
    h = entropy(probabilities)
    assert abs(sum(probabilities) - 1) < 1e-12
    assert kraft <= 1 + 1e-12
    assert h <= mean + 1e-12
    assert mean < h + 1 + 1e-12
    return h, mean, kraft


assert huffman_lengths([0, 0.25, 0.25, 0.5]) == {3: 1, 1: 2, 2: 2}
assert abs(inspect([0, 0.25, 0.25, 0.5])[1] - 1.5) < 1e-12

for name, p in [
    ("four-symbol", [1/2, 1/4, 1/8, 1/8]),
    ("eight-symbol", [1/2, 1/4, 1/8, 1/16] + [1/64]*4),
]:
    h, mean, kraft = inspect(p)
    print(f"{name}: H={h:.6f}, L={mean:.6f}, Kraft={kraft:.6f}")

print("iid binary source p=(0.9, 0.1)")
base = [0.9, 0.1]
for block in range(1, 6):
    p = [math.prod(base[j] for j in word)
         for word in itertools.product(range(2), repeat=block)]
    h, mean, _ = inspect(p)
    assert abs(h/block - entropy(base)) < 1e-12
    print(f"m={block}: H/m={h/block:.6f}, L/m={mean/block:.6f}, "
          f"excess={mean/block - entropy(base):.6f}")

print(f"rain surprise={-math.log2(0.005):.6f} bits")
print(f"no-rain surprise={-math.log2(0.995):.6f} bits")
print(f"fair die H={entropy([1/6]*6):.6f} bits")
print(f"biased die H={entropy([0.1]*5 + [0.5]):.6f} bits")

permutation = []
for a, b, c in itertools.product(range(2), repeat=3):
    index = 4*a + 2*b + c
    result = 4*a + 2*b + (c ^ (a & b))
    permutation.append(result)
    assert (c ^ (a & b)) ^ (a & b) == c
assert sorted(permutation) == list(range(8))
print("Toffoli permutation:", permutation)
