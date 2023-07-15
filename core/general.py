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
    "R_k": 25812.807 ##sp.constants.h/ sp.constants.e**2.0,  # Normal quantum resistance, aka Klitzing constant.
    # Note, in some papers a superconducting quantum
    # resistance is used, and defined as: h/(2e)^2
}

#Useful Functions 
def process_op(native_op):
        ##### transform operator from native basis to energy eigenbasis 
    return native_op

def calc_therm_ratio(
    omega: float, 
    T: float = NOISE_PARAMS["T"]
    ):
    return (sp.constants.hbar * omega) / (sp.constants.k * T)


#Useful Operators
def annihilation(
          dimension: int):
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)

def creation(
          dimension: int):
    return annihilation(dimension).T



#General T1 and Tphi Functions
def t1_rate(
        noise_op: torch.Tensor,
        spectral_density: torch.Tensor,
        eigvecs: torch.Tensor,
    ) -> torch.Tensor:
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


def tphi_rate(
        A_noise: float,
        noise_op: torch.Tensor,
        eigvecs: torch.Tensor, 
        omega_low: float = NOISE_PARAMS["omega_low"], 
        t_exp: float = NOISE_PARAMS["t_exp"]
    ) -> torch.Tensor:
        
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


#Effective T1 and Tphi Functions
def effective_t1_rate(
          qubit, 
          noise_channels: Union[str, List[str]],
          T: float = NOISE_PARAMS['T'],
          M: float = NOISE_PARAMS['M'],
          R_0: float = NOISE_PARAMS['R_0'],
          R_k: float = NOISE_PARAMS['R_k']
          ):

    eigvals,eigvecs = qubit.esys()

    t1 = torch.zeros([1,1],dtype=torch.double)

    if "t1_capacitive" in noise_channels:
         t1 += t1_rate(noise_op = qubit.n_operator(), spectral_density = spectral_density_cap(qubit, T), eigvecs = eigvecs)

    if "t1_flux_bias_line" in noise_channels:
         t1 += t1_rate(noise_op = qubit.d_hamiltonian_d_flux(), spectral_density = spectral_density_fbl(qubit), eigvecs = eigvecs)

    if "t1_charge_impedance" in noise_channels:
         t1 += t1_rate(noise_op = qubit.n_operator(), spectral_density = spectral_density_ci(qubit), eigvecs = eigvecs)
  
    if "t1_inductive" in noise_channels:
         t1 += t1_rate(noise_op = qubit.phi_operator()+0j, spectral_density = spectral_density_ind(qubit), eigvecs = eigvecs)

    if "t1_quasiparticle_tunneling" in noise_channels:
         t1 += t1_rate(noise_op = qt_noise_op(qubit)+0j, spectral_density = spectral_density_qt(qubit), eigvecs = eigvecs)

    return t1


def effective_tphi_rate(
          qubit , 
          noise_channels: Union[str, List[str]] ,
          A_cc: float = NOISE_PARAMS["A_cc"], 
          A_flux: float = NOISE_PARAMS["A_flux"]
          ):

    eigvals,eigvecs = qubit.esys()

    tphi = torch.zeros([1,1],dtype=torch.double)

    #tphi_1_over_f_flux
    if 'tphi_1_over_f_flux' in noise_channels:
         tphi += tphi_rate(A_flux, qubit.d_hamiltonian_d_flux(),  eigvecs=eigvecs)
    
    #tphi_1_over_f_cc
    if 'tphi_1_over_f_cc' in noise_channels:
         tphi += tphi_rate(A_cc, qubit.d_hamiltonian_d_EJ(), eigvecs=eigvecs)

    return  tphi

#Union[ZeroPi, Fluxonium]
def t2_rate(qubit , 
       t1_noise_channels: Union[str, List[str]],
       tphi_noise_channels: Union[str, List[str]]
       ):
        t1_rate = effective_t1_rate(qubit, noise_channels = t1_noise_channels)
        tphi_rate = effective_tphi_rate(qubit, noise_channels = tphi_noise_channels)
        return 0.5*t1_rate + tphi_rate


#Spectral Densities for T1 Calcs

#T1 CAPACITIVE FUNCTIONS
def q_cap_fun(qubit) -> torch.Tensor:
    return (
        1e6
        * torch.pow((2 * np.pi * 6e9 / torch.abs(qubit.omega())) , 0.7)
    )

def spectral_density_cap(qubit, 
                         T: float = NOISE_PARAMS['T']):
    
    therm_ratio = calc_therm_ratio(qubit.omega(), T)
    s = (
        2
        * 8
        * qubit.EC
        / q_cap_fun(qubit)
        * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
        / (1 + torch.exp(-therm_ratio))
    )
    s *= (
        2 * np.pi
    )  # We assume that system energies are given in units of frequency
    return s



#T1 FLUX BIAS LINE FUNCTIONS
def spectral_density_fbl(
        qubit,
        M: float = NOISE_PARAMS["M"], 
        R_0: float = NOISE_PARAMS["R_0"], 
        T: float = NOISE_PARAMS["T"]):
        
        therm_ratio = calc_therm_ratio(qubit.omega(), T)
        s = (
            2
            * (2 * np.pi) ** 2
            * M**2
            * (qubit.omega() / 1e9 )
            * sp.constants.hbar
            / complex(R_0).real
            * (1 / torch.tanh(0.5 * therm_ratio))
            / (1 + torch.exp(-therm_ratio))
            )
        s *= (2 * np.pi)  # We assume that system energies are given in units of frequency
        return s

#T1 CHARGE IMPEDANCE FUNCTIONS 
def spectral_density_ci(
          qubit, 
          R_0: float = NOISE_PARAMS["R_0"], 
          T: float = NOISE_PARAMS["T"], 
          R_k: float = NOISE_PARAMS["R_k"]
          ):
    # Note, our definition of Q_c is different from Zhang et al (2020) by a
    # factor of 2

    Q_c = R_k / (8 * np.pi * complex(R_0).real)
    therm_ratio = calc_therm_ratio(qubit.omega(), T)
    s = (
        2
        * (qubit.omega() / 1e9 ) #annoying unit stuff - gotta convert back to GHz
        / Q_c
        * (1 / torch.tanh(0.5 * therm_ratio))
        / (1 + torch.exp(-therm_ratio))
    )
    return s

#T1 INDUCTIVE
def q_ind_fun(
            qubit, 
            T: float = NOISE_PARAMS["T"]
            ):
    therm_ratio = abs(calc_therm_ratio(qubit.omega(), T))
    therm_ratio_500MHz = calc_therm_ratio(
        omega =  torch.tensor(2 * np.pi * 500e6), T=T
    )
    return (
        500e6
        * (
            torch.special.scaled_modified_bessel_k0(1 / 2 * therm_ratio_500MHz)
            * torch.sinh(1 / 2 * therm_ratio_500MHz)
            / torch.exp(1 / 2 * therm_ratio_500MHz)
        )
        / (
            torch.special.scaled_modified_bessel_k0(1 / 2 * therm_ratio)
            * torch.sinh(1 / 2 * therm_ratio)
            / torch.exp(1 / 2 * therm_ratio)
        )
    )

def spectral_density_ind(
            qubit, 
            T: float = NOISE_PARAMS["T"]
            ):
        therm_ratio = calc_therm_ratio(qubit.omega(),T)
        s = (
            2
            * qubit.EL
            / q_ind_fun(qubit, T)
            * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
            / (1 + torch.exp(-therm_ratio))
        )
        s *= (
            2 * np.pi
        )  # We assume that system energies are given in units of frequency
        return s

# T1 QUASIPARTICLE TUNNELLING FUNCTIONS

def y_qp_fun(
          qubit, 
          T: float = NOISE_PARAMS["T"], 
          R_k: float = NOISE_PARAMS["R_k"], 
          Delta: float = NOISE_PARAMS['Delta'],
          x_qp: float = NOISE_PARAMS['x_qp']):
    
    omega = torch.abs(qubit.omega())
    Delta_in_Hz = Delta * sp.constants.e / sp.constants.h
    omega_in_Hz = omega / (2 * np.pi) 
    EJ_in_Hz = qubit.EJ * 1e9 # GHz to Hz

    therm_ratio = calc_therm_ratio(qubit.omega(), T)
    re_y_qp = (
        np.sqrt(2 / np.pi)
        * (8 / R_k)
        * (EJ_in_Hz / Delta_in_Hz)
        * (2 * Delta_in_Hz / omega_in_Hz) ** (3 / 2)
        * x_qp
        * torch.sqrt(1 / 2 * therm_ratio)
        * torch.special.scaled_modified_bessel_k0(1 / 2 * torch.abs(therm_ratio))
        * torch.sinh(1 / 2 * therm_ratio)
        / torch.exp(1 / 2 * torch.abs(therm_ratio))
    )

    return re_y_qp

def spectral_density_qt(
          qubit, 
          T: float = NOISE_PARAMS["T"],
          ):
    therm_ratio = calc_therm_ratio(qubit.omega(), T)
    return (
        2
        * qubit.omega() / 1e9
        * complex(y_qp_fun(qubit)).real
        * (1 / torch.tanh(0.5 * therm_ratio))
        / (1 + torch.exp(-therm_ratio))
    )
    
def qt_noise_op(qubit):
    argument = qubit.phi_operator() + 2 * np.pi * qubit.flux * torch.tensor(np.eye(qubit.dim),dtype=torch.double)
    qt_argument = argument/2
    return (torch.linalg.matrix_exp(qt_argument*1j)-torch.linalg.matrix_exp(qt_argument*-1j))/2j



