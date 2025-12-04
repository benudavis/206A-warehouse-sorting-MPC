"""CasADi wrapper for neural network forward kinematics."""

import numpy as np
import casadi as ca
from pathlib import Path


def build_nn_fk_function(weights_path: str, n_joints: int = 6) -> ca.Function:
    """Build CasADi symbolic function nn_fk(q) -> end_effector_position."""
    params = np.load(str(Path(weights_path)))
    
    W1, b1 = ca.DM(params["W1"]), ca.DM(params["b1"])
    W2, b2 = ca.DM(params["W2"]), ca.DM(params["b2"])
    W3, b3 = ca.DM(params["W3"]), ca.DM(params["b3"])
    W4, b4 = ca.DM(params["W4"]), ca.DM(params["b4"])
    
    mu_phi = ca.DM(params["mu_q"])
    sigma_phi = ca.DM(params["sigma_q"])
    mu_p = ca.DM(params["mu_p"])
    sigma_p = ca.DM(params["sigma_p"])
    
    q = ca.SX.sym("q", n_joints)
    phi = ca.vertcat(ca.sin(q), ca.cos(q))
    phi_norm = (phi - mu_phi) / sigma_phi
    
    h1 = ca.tanh(W1 @ phi_norm + b1)
    h2 = ca.tanh(W2 @ h1 + b2)
    h3 = ca.tanh(W3 @ h2 + b3)
    p_norm = W4 @ h3 + b4
    p = sigma_p * p_norm + mu_p
    
    return ca.Function("nn_fk", [q], [p], ["q"], ["p"])

