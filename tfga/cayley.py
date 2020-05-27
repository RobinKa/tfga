"""Operations for constructing the cayley 3-tensor needed
for the geometric product. Used internally.
"""
from itertools import combinations
import numpy as np

from .blades import get_normal_ordered


def _collapse_same(x):
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a == b:
            return False, x[:i] + x[i+2:], a
    return True, x, None


def _reduce_bases(a, b, metric):
    if a == "":
        return 1, b
    elif b == "":
        return 1, a

    combined = list(a + b)

    # Bring into normal order:
    sign, combined = get_normal_ordered(combined)

    done = False
    while not done:
        done, combined, combined_elem = _collapse_same(combined)
        if not done:
            sign *= metric[combined_elem]

    return sign, "".join(combined)


def blades_from_bases(vector_bases):
    all_combinations = [""]
    degrees = [0]
    for i in range(1, len(vector_bases) + 1):
        combs = combinations(vector_bases, i)
        combs = ["".join(c) for c in combs]
        all_combinations += combs
        degrees += [i] * len(combs)
    return all_combinations, degrees


def get_cayley_tensor(metric, bases, blades):
    dims = len(metric)
    num_blades = len(blades)

    t_geom = np.zeros((num_blades, num_blades, num_blades), dtype=np.int32)
    t_inner = np.zeros((num_blades, num_blades, num_blades), dtype=np.int32)
    t_outer = np.zeros((num_blades, num_blades, num_blades), dtype=np.int32)

    metric_dict = {v: metric[i] for i, v in enumerate(bases)}

    for a in blades:
        for b in blades:
            sign, result = _reduce_bases(a, b, metric_dict)
            a_index = blades.index(a)
            b_index = blades.index(b)
            out_index = blades.index(result)
            t_geom[a_index, b_index, out_index] = sign

            # Degree went down -> part of inner
            if len(result) == abs(len(a) - len(b)):
                t_inner[a_index, b_index, out_index] = sign

            # Degree went up -> part of outer
            if len(result) == len(a) + len(b):
                t_outer[a_index, b_index, out_index] = sign

    return t_geom, t_inner, t_outer
