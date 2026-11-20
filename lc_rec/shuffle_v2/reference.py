"""Independent implementation of the V2 draw and destination mapping for checks."""
from hashlib import sha256
from math import factorial


def relative(m, row, occurrence, hash_bytes=sha256):
    size = factorial(m)
    counter = 0
    while True:
        key = f'["GRID-LETTER-SHUFFLE-V2",42,{m:d},{row:d},{occurrence:d},{counter:d}]'
        number = int(hash_bytes(key.encode('utf-8')).hexdigest(), 16)
        if (1 << 256) - number > (1 << 256) % size:
            break
        counter += 1
    index = number % size
    residual, pool, output = index, list(range(m)), []
    for place in range(m - 1, -1, -1):
        quotient, residual = divmod(residual, factorial(place))
        output.append(pool.pop(quotient))
    return index, output, counter


def remap(original, active, permutation):
    replacements = {dst: original[active[src]] for dst, src in zip(active, permutation, strict=True)}
    return [replacements.get(pos, token) for pos, token in enumerate(original)]
