import torch
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

NOISE_PARAMS = {
    "A_flux": 1e-6,  # Flux noise strength. Units: Phi_0
    "A_ng": 1e-4,  # Charge noise strength. Units of charge e
    "A_cc": 1e-7,  # Critical current noise strength. Units of critical current I_c
    "omega_low": 1e-9 * 2 * np.pi,  # Low frequency cutoff. Units: 2pi GHz
    "omega_high": 3 * 2 * np.pi,  # High frequency cutoff. Units: 2pi GHz
    "Delta": 3.4e-4,  # Superconducting gap for aluminum (at T=0). Units: eV
    "x_qp": 3e-6,  # Quasiparticles density (see for example Pol et al 2014)
    "t_exp": 1e4,  # Measurement time. Units: ns
    "R_0": 50,  # Characteristic impedance of a transmission line. Units: Ohms
    "T": 0.015,  # Typical temperature for a superconducting circuit experiment. Units: K
    "M": 400,  # Mutual inductance between qubit and a flux line. Units: \Phi_0 / Ampere
    "R_k": sp.constants.h
    / sp.constants.e**2.0,  # Normal quantum resistance, aka Klitzing constant.
    # Note, in some papers a superconducting quantum
    # resistance is used, and defined as: h/(2e)^2
}


def process_op(native_op):
        ##### transform operator from native basis to energy eigenbasis 
    return native_op

def calc_therm_ratio(
    omega: float, T: float = NOISE_PARAMS["T"]
    ):
    return (sp.constants.hbar * omega) / (sp.constants.k * T)


def annihilation(dimension: int):
    """
    Returns a dense matrix of size dimension x dimension representing the annihilation
    operator in number basis.
    """
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)

def creation(dimension: int):
    """
    Returns a dense matrix of size dimension x dimension representing the creation
    operator in number basis.
    """
    return annihilation(dimension).T

def t1(
        noise_op: torch.Tensor,
        spectral_density: torch.Tensor,
        eigvecs: torch.Tensor,
    ):
        # We assume that the energies in `evals` are given in the units of frequency
        # and *not* angular frequency. The function `spectral_density` is assumed to
        # take as a parameter an angular frequency, hence we have to convert.

        ground = eigvecs[:,0]
        excited = eigvecs[:,1]
        
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
        eigvecs: torch.Tensor
    ) -> float:
        
        omega_low = 1e-9 * 2 * np.pi  # Low frequency cutoff. Units: 2pi GHz
        t_exp = 1e4  #Measurement time. Units: ns

        ground = eigvecs[:,0]
        excited = eigvecs[:,1]
        

        rate = torch.abs(
            torch.matmul(ground.conj().to(torch.complex128),torch.matmul(noise_op.to(torch.complex128),ground.T.to(torch.complex128)))
            - torch.matmul(excited.conj().to(torch.complex128),torch.matmul(noise_op.to(torch.complex128),excited.T.to(torch.complex128)))
        )

        rate *=  A_noise * np.sqrt(2 * np.abs(np.log(omega_low * t_exp)))

        # We assume that the system energies are given in units of frequency and
        # not the angular frequency, hence we have to multiply by `2\pi`
        rate *= 2 * np.pi

        return rate


def effective_t1_time(qubit):

    eigvals,eigvecs = qubit.esys()

    t1_rate = torch.zeros([1,1],dtype=torch.double)

    #t1_capacitive

    t1_rate += t1(noise_op = qubit.n_op(), spectral_density = qubit.spectral_density_cap(T), eigvecs = eigvecs)

    #charge impendenace not included?

    #t1_flux_bias_line

    t1_rate += t1(noise_op = qubit.d_ham_d_flux(), spectral_density = qubit.spectral_density_fbl(T, M, R_0), eigvecs = eigvecs)

    #3.4 t1_inductive

    t1_rate += t1(noise_op = qubit.phi()+0j, spectral_density = qubit.spectral_density_ind(T), eigvecs = eigvecs)

    #3.5 t1_quasiparticle_tunneling - a little dubious

    t1_rate += t1(noise_op = qubit.qt_noise_op()+0j, spectral_density = qubit.spectral_density_qt(T, R_k), eigvecs = eigvecs)

    return t1_rate


def effective_tphi_time(qubit, A_cc, A_flux, noise_channels):

    eigvals,eigvecs = qubit.esys()

    tphi_rate = torch.zeros([1,1],dtype=torch.double)

    #tphi_1_over_f_flux
    if 'tphi_1_over_f_flux' in noise_channels:
         tphi_rate += tphi(A_flux, qubit.d_ham_d_flux(),  eigvecs=eigvecs)
    
    #tphi_1_over_f_cc
    if 'tphi_1_over_f_cc' in noise_channels:
         tphi_rate += tphi(A_cc, qubit.d_ham_d_EJ(), eigvecs=eigvecs)

    return  tphi_rate


def t2(qubit):
    
    return 1/(0.5*effective_t1_time(qubit))+1/effective_tphi_time(qubit)



