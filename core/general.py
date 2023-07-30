import torch
import numpy as np
import scipy as sp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import scipy.constants

# NOISE_PARAMS TAKEN FROM SCQUBITS
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
    "R_k": sp.constants.h / (sp.constants.e**2.0)  # Normal quantum resistance, aka Klitzing constant.
    # Note, in some papers a superconducting quantum
    # resistance is used, and defined as: h/(2e)^2
}


# USEFUL FUNCTIONS
def change_operator_basis(
    operator: torch.Tensor,
    change_of_basis_matrix: torch.Tensor,
) -> torch.Tensor:
    operator = (
        operator.to(torch.complex128)
        if change_of_basis_matrix.dtype == torch.complex128
        else operator.to(torch.float64)
    )
    new_operator = change_of_basis_matrix.conj().T @ operator @ change_of_basis_matrix
    return new_operator


def calc_therm_ratio(omega: torch.Tensor, T: float = NOISE_PARAMS["T"]):
    # omega must be in units of radians/s
    return (sp.constants.hbar * omega * 1e9) / (sp.constants.k * T)


# USEFUL OPERATORS
def annihilation(dimension: int) -> np.ndarray:
    offdiag_elements = np.sqrt(range(1, dimension))
    return np.diagflat(offdiag_elements, 1)


def creation(dimension: int) -> np.ndarray:
    return annihilation(dimension).T


def omega(qubit) -> torch.Tensor:
    # angular frequency between ground and excited states
    eigvals = qubit.esys()[0]
    ground_E = eigvals[0]
    excited_E = eigvals[1]
    return 2 * np.pi * (excited_E - ground_E)


# GENERIC T1 AND TPHI FORMULAS
def t1_rate(
    noise_op: torch.Tensor,
    spectral_density: torch.Tensor,
    eigvecs: torch.Tensor,
) -> torch.Tensor:
    ground = eigvecs[:, 0]
    excited = eigvecs[:, 1]

    s = spectral_density
    ##Note the distinction in code where they have spectral_density(omega)+spectral_density(-omega)

    rate = torch.matmul(noise_op.to(torch.complex128), ground.T.to(torch.complex128))
    rate = torch.matmul(excited.conj().to(torch.complex128), rate)
    rate = torch.pow(torch.abs(rate), 2) * s
    rate = rate

    # Returns rate in 1/(2pi*1e9) s --> or maybe not? maybve just 1e9

    return rate


def tphi_rate(
    A_noise: float,
    noise_op: torch.Tensor,
    eigvecs: torch.Tensor,
    omega_low: float = NOISE_PARAMS["omega_low"],
    t_exp: float = NOISE_PARAMS["t_exp"],
) -> torch.Tensor:
    ground = eigvecs[:, 0]
    excited = eigvecs[:, 1]

    rate = torch.abs(
        torch.matmul(
            ground.conj().to(torch.complex128),
            torch.matmul(noise_op.to(torch.complex128), ground.T.to(torch.complex128)),
        )
        - torch.matmul(
            excited.conj().to(torch.complex128),
            torch.matmul(noise_op.to(torch.complex128), excited.T.to(torch.complex128)),
        )
    )
    rate *= A_noise * np.sqrt(2 * np.abs(np.log(omega_low * t_exp)))

    # We assume that the system energies are given in units of frequency and
    # not the angular frequency, hence we have to multiply by `2\pi`
    rate *= 2 * np.pi

    return rate


# T1 AND TPHI ACROSS SPECIFIC NOISE CHANNELS
def effective_t1_rate(
    qubit,
    noise_channels: Union[str, List[str]],
    T: float = NOISE_PARAMS["T"],
    M: float = NOISE_PARAMS["M"],
    R_0: float = NOISE_PARAMS["R_0"],
    R_k: float = NOISE_PARAMS["R_k"],
) -> torch.Tensor:
    t1 = torch.zeros([1, 1], dtype=torch.double)

    if "t1_capacitive" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.n_operator(),
            spectral_density=spectral_density_cap(qubit, True, T) + spectral_density_cap(qubit, False, T),
            eigvecs=qubit.esys()[1],
        )

    if "t1_flux_bias_line" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.d_hamiltonian_d_flux(),
            spectral_density=spectral_density_fbl(qubit, True, M, R_0, T)
            + spectral_density_fbl(qubit, False, M, R_0, T),
            eigvecs=qubit.esys()[1],
        )

    if "t1_charge_impedance" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.n_operator(),
            spectral_density=spectral_density_ci(qubit, True, R_0, T, R_k)
            + spectral_density_ci(qubit, False, R_0, T, R_k),
            eigvecs=qubit.esys()[1],
        )

    if "t1_inductive" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.phi_operator(),
            spectral_density=spectral_density_ind(qubit, True, T) + spectral_density_ind(qubit, False, T),
            eigvecs=qubit.esys()[1],
        )

    if "t1_quasiparticle_tunneling" in noise_channels:
        t1 += t1_rate(
            noise_op=qubit.sin_phi_operator(alpha=0.5, beta=0.5 * (2 * np.pi * qubit.flux)),
            spectral_density=spectral_density_qt(qubit, True, T) + spectral_density_qt(qubit, False, T),
            eigvecs=qubit.esys()[1],
        )

    return t1


def effective_tphi_rate(
    qubit,
    noise_channels: Union[str, List[str]],
    A_cc: float = NOISE_PARAMS["A_cc"],
    A_flux: float = NOISE_PARAMS["A_flux"],
) -> torch.Tensor:
    eigvecs = qubit.esys()[1]
    tphi = torch.zeros([1, 1], dtype=torch.double)

    # tphi_1_over_f_flux
    if "tphi_1_over_f_flux" in noise_channels:
        tphi += tphi_rate(A_flux, qubit.d_hamiltonian_d_flux(), eigvecs=eigvecs)

    # tphi_1_over_f_cc
    if "tphi_1_over_f_cc" in noise_channels:
        tphi += tphi_rate(A_cc, qubit.d_hamiltonian_d_EJ(), eigvecs=eigvecs)

    return tphi


# T2 RATE
def t2_rate(
    qubit,  # Union[ZeroPi, Fluxonium],
    t1_noise_channels: Union[str, List[str]],
    tphi_noise_channels: Union[str, List[str]],
) -> torch.Tensor:
    t1_rate = effective_t1_rate(qubit, noise_channels=t1_noise_channels)
    tphi_rate = effective_tphi_rate(qubit, noise_channels=tphi_noise_channels)
    return 0.5 * t1_rate + tphi_rate


# SPECTRAL DENSITIES FOR T1


# CAPACITIVE
def q_cap_fun(qubit) -> torch.Tensor:
    return 1e6 * torch.pow((2 * np.pi * 6e9 / torch.abs(omega(qubit) * (1e9))), 0.7)


def spectral_density_cap(qubit, plus_minus_omega: bool, T: float = NOISE_PARAMS["T"]):
    omega_for_calc = omega(qubit) if plus_minus_omega else -omega(qubit)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    s = (
        2
        * 8
        * qubit.EC
        / q_cap_fun(qubit)
        * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
        / (1 + torch.exp(-therm_ratio))
    )
    s *= 2 * np.pi  # We assume that system energies are given in units of frequency
    return s


# FLUX BIAS LINE
def spectral_density_fbl(
    qubit,
    plus_minus_omega: bool,
    M: float = NOISE_PARAMS["M"],
    R_0: float = NOISE_PARAMS["R_0"],
    T: float = NOISE_PARAMS["T"],
):
    omega_for_calc = omega(qubit) if plus_minus_omega else -omega(qubit)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    s = (
        2
        * (2 * np.pi) ** 2
        * M**2
        * (omega(qubit) * 1e9)
        * sp.constants.hbar
        / R_0
        * (1 / torch.tanh(0.5 * therm_ratio))
        / (1 + torch.exp(-therm_ratio))
    )

    # Unsure why an extra factor of 1e9 is needed?
    return s * 1e9


# CHARGE IMPEDANCE
def spectral_density_ci(
    qubit,
    plus_minus_omega: bool,
    R_0: float = NOISE_PARAMS["R_0"],
    T: float = NOISE_PARAMS["T"],
    R_k: float = NOISE_PARAMS["R_k"],
):
    # Note, our definition of Q_c is different from Zhang et al (2020) by a
    # factor of 2

    omega_for_calc = omega(qubit) if plus_minus_omega else -omega(qubit)

    Q_c = R_k / (8 * np.pi * complex(R_0).real)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    s = 2 * (omega_for_calc) / Q_c * (1 / torch.tanh(0.5 * therm_ratio)) / (1 + torch.exp(-therm_ratio))
    return s


# INDUCTIVE
def q_ind_fun(qubit, plus_minus_omega: bool, T: float = NOISE_PARAMS["T"]):
    omega_for_calc = omega(qubit) if plus_minus_omega else -omega(qubit)
    therm_ratio = abs(calc_therm_ratio(omega_for_calc, T))
    therm_ratio_500MHz = calc_therm_ratio(omega=torch.tensor(2 * np.pi * 500e6) / 1e9, T=T)

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
    )  ##multiplying each through by the torch.exp(1 / 2 * therm_ratio) seems to work but is different to scqubits?


def spectral_density_ind(qubit, plus_minus_omega: bool, T: float = NOISE_PARAMS["T"]):
    omega_for_calc = omega(qubit) if plus_minus_omega else -omega(qubit)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)

    s = (
        2
        * qubit.EL
        / q_ind_fun(qubit, T)
        * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
        / (1 + torch.exp(-therm_ratio))
    )
    s *= 2 * np.pi  # We assume that system energies are given in units of frequency
    return s


# QUASIPARTICLE TUNNELLING
def y_qp_fun(
    qubit,
    T: float = NOISE_PARAMS["T"],
    R_k: float = NOISE_PARAMS["R_k"],
    Delta: float = NOISE_PARAMS["Delta"],
    x_qp: float = NOISE_PARAMS["x_qp"],
):
    Delta_in_Hz = Delta * sp.constants.e / sp.constants.h
    omega_in_Hz = torch.abs(omega(qubit)) * 1e9 / (2 * np.pi)
    EJ_in_Hz = qubit.EJ * 1e9

    therm_ratio = calc_therm_ratio(torch.abs(omega(qubit)), T)
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
    plus_minus_omega: bool,
    T: float = NOISE_PARAMS["T"],
):
    omega_for_calc = omega(qubit) if plus_minus_omega else -omega(qubit)
    therm_ratio = calc_therm_ratio(omega_for_calc, T)
    return (
        2
        * omega_for_calc
        * complex(y_qp_fun(qubit)).real
        * (1 / torch.tanh(0.5 * therm_ratio))
        / (1 + torch.exp(-therm_ratio))
    )
