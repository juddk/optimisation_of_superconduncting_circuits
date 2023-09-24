import autograd
import numpy as np
import scipy.sparse as sp
from autograd.extend import primitive, defvjp, vspace
import autograd.numpy as npa

""" Define here various primitives needed for the main code 
To use with both numpy and autograd backends, define the autograd primitive of 
a numpy function fnc as fnc_ag, and then define the vjp"""

def T(x): return np.swapaxes(x, -1, -2)

eigsh_ag = primitive(sp.linalg.eigsh)

def vjp_maker_eigsh(ans, x, numeig=10, sigma=0.):
    """Gradient for eigenvalues and vectors of a hermitian matrix."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    vc = np.conj(v)
    
    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = np.repeat(w[..., np.newaxis], numeig, axis=-1)

        # Eigenvalue part
        vjp_temp = np.dot(vc * wg[..., np.newaxis, :], T(v)) 

        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function 
        # depends on the eigenvalues only.
        if np.any(vg):
            off_diag = np.ones((numeig, numeig)) - np.eye(numeig)
            F = off_diag / (T(w_repeated) - w_repeated + np.eye(numeig))
            vjp_temp += np.dot(np.dot(vc, F * np.dot(T(v), vg)), T(v))

        return vjp_temp

    return vjp

defvjp(eigsh_ag, vjp_maker_eigsh)