import torch
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


def process_op(native_op, energy_esys, truncated_dim):
        ##### transform operator from native basis to energy eigenbasis 
    return native_op

def calc_therm_ratio(
    omega: float, T: float
    ):
    return (sp.constants.hbar * omega) / (sp.constants.k * T)

def t1(
        noise_op: torch.Tensor,
        spectral_density: torch.Tensor,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
        EL: torch.Tensor
        
    ):
        # We assume that the energies in `evals` are given in the units of frequency
        # and *not* angular frequency. The function `spectral_density` is assumed to
        # take as a parameter an angular frequency, hence we have to convert.

        ground = eigvecs[:,0]
        excited = eigvecs[:,1]
        ground_E = eigvals[0]
        excited_E = eigvals[1]

        omega = 2 * np.pi * (excited_E - ground_E) * 1e9
        
        s = spectral_density 
        ##Note the distinction in code where they have spectral_density(omega)+spectral_density(-omega) may
        #need to add that in zero pi module?
    
        rate = torch.matmul(noise_op.to(torch.complex128),ground.T.to(torch.complex128))
        rate = torch.matmul(excited.conj().to(torch.complex128),rate)
        rate = torch.pow(torch.abs(rate) , 2) * s

        return rate


def tphi(
        A_noise: float,
        noise_op: torch.Tensor,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
        omega_low: float,
        t_exp: float
    ) -> float:

        ground = eigvecs[:,0]
        excited = eigvecs[:,1]
        ground_E = eigvals[0]
        excited_E = eigvals[1]

        rate = torch.abs(
            torch.matmul(ground.conj().to(torch.complex128),torch.matmul(noise_op.to(torch.complex128),ground.T.to(torch.complex128)))
            - torch.matmul(excited.conj().to(torch.complex128),torch.matmul(noise_op.to(torch.complex128),excited.T.to(torch.complex128)))
        )

        rate *=  A_noise * np.sqrt(2 * np.abs(np.log(omega_low * t_exp)))

        # We assume that the system energies are given in units of frequency and
        # not the angular frequency, hence we have to multiply by `2\pi`
        rate *= 2 * np.pi

        return rate


