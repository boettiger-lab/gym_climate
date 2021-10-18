import numpy as np

current_state = [240, 7e13, 5e11]

# rescaling parameters
A_mid = current_state[0]
W_mid = current_state[1]
S_mid = current_state[2]


def ays_rescaled_rhs(
    ays,
    t=0,
    beta=None,
    epsilon=None,
    phi=None,
    rho=None,
    sigma=None,
    tau_A=None,
    tau_S=None,
    theta=None,
):
    # This system makes the transformation X = X_mid * x / (1 - x) where x is
    # in [0, 1]. This scale is more conducive to ML.
    a, y, s = ays
    s_inv = 1 - s
    s_inv_rho = s_inv ** rho
    K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho)

    a_inv = 1 - a
    w_inv = 1 - y
    Y = W_mid * y / w_inv
    A = A_mid * a / a_inv
    adot = K / (phi * epsilon * A_mid) * a_inv * a_inv * Y - a * a_inv / tau_A
    ydot = y * w_inv * (beta - theta * A)
    sdot = (1 - K) * s_inv * s_inv * Y / (epsilon * S_mid) - s * s_inv / tau_S

    return adot, ydot, sdot
