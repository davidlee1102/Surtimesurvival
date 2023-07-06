import math

from fractions import Fraction


def finding_time_t(u, k, g, c):
    if not isinstance(k, int) and k != 0:
        k = Fraction.from_float(k).limit_denominator()

    exp_c = math.exp(c)  # Calculate exp(c) once and store the result
    ln_value = math.log(1 / (1 - u))
    right_hand_side = ln_value / exp_c

    if right_hand_side >= 0:
        t = g * right_hand_side ** (1 / k)
        return t
    else:
        return None  # Return None to indicate no solution
