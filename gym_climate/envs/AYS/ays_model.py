import numpy as np

def ays_rhs(ays, t, beta, epsilon, phi, rho, sigma, theta, tau_A, tau_S):
  A, Y, S = ays
  Gamma = 1 / (1 + (S / sigma)**rho)
  U = Y / epsilon
  R = (1 - Gamma) * U
  F = Gamma * U
  E = F / phi
  
  a_dot = E - A / tau_A
  y_dot = (beta - theta * A) * Y
  s_dot = R - S / tau_S
  return a_dot, y_dot, s_dot
