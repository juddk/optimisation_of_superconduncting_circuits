import torch
import general
import scqubits as sc
import numpy as np
import math
import scipy as sp


class Fluxonium:
    def __init__(self,
                 EJ: torch.Tensor,
                 EC: torch.Tensor,
                 EL: torch.Tensor,
                 flux:torch.Tensor,
                 dim: int, 
                 hamiltonian_creation: str):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux = flux 
        self.dim = dim
        self.hamiltonian_creation = hamiltonian_creation


    def phi_operator(self):
          phi_osc = torch.pow((8.0 * self.EC/self.EL) , 0.25)
          phi = (
                (torch.tensor(general.creation(self.dim),dtype=torch.double) + torch.tensor(general.annihilation(self.dim),dtype=torch.double))
                * phi_osc/ math.sqrt(2)
            )
          return phi 

    def cos_phi_operator(self):
        argument = self.phi_operator() + 2 * np.pi * self.flux * torch.tensor(np.eye(self.dim),dtype=torch.double)
        cos_phi = (torch.linalg.matrix_exp(argument*1j)+torch.linalg.matrix_exp(argument*-1j))/2
        return cos_phi
    
    def sin_phi_operator(self):
        argument = self.phi_operator() + 2 * np.pi * self.flux * torch.tensor(np.eye(self.dim),dtype=torch.double)
    
        sin_phi = (torch.linalg.matrix_exp(argument*1j)-torch.linalg.matrix_exp(argument*-1j))/2j

        return sin_phi 
    
    
    def n_operator(self):
        
        phi_osc = torch.pow((8.0 * self.EC/ self.EL) , 0.25)
        n_op = (
                1j
                * (torch.tensor(general.creation(self.dim),dtype=torch.double) - torch.tensor(general.annihilation(self.dim),dtype=torch.double))
                / (phi_osc * math.sqrt(2))
            )
        return n_op
    
    def d_hamiltonian_d_flux(self):
        
        return -2 * np.pi * self.EJ * self.sin_phi_operator()
    

    def d_hamiltonian_d_EJ(self):
        return -self.cos_phi_operator()
    

    def create_H(self):
        plasma_energy = torch.sqrt(8.0*self.EL * self.EC)
        diag_elements = [(i + 0.5) for i in range(self.dim)]
        lc_osc = torch.tensor(np.diag(diag_elements),dtype=torch.double)
        lc_osc = lc_osc * plasma_energy
        
        return lc_osc - self.EJ*self.cos_phi_operator()

    
    def auto_H(self):
        return torch.from_numpy(
            sc.Fluxonium(EJ=self.EJ.item(), EC=self.EC.item(), EL = self.EL.item(), flux = self.flux.item(), cutoff= self.dim).hamiltonian())
    

    def t1_supported_noise_channels(self):
        t1_supported_noise_channels = []
        for x in sc.Fluxonium(EJ=self.EJ.item(), EC=self.EC.item(), EL = self.EL.item(), flux = self.flux.item(), cutoff= self.dim).supported_noise_channels():
            if x.startswith("t1"):
                t1_supported_noise_channels.append(x)
        return t1_supported_noise_channels

    def tphi_supported_noise_channels(self):
        tphi_supported_noise_channels = []
        for x in sc.Fluxonium(EJ=self.EJ.item(), EC=self.EC.item(), EL = self.EL.item(), flux = self.flux.item(), cutoff= self.dim).supported_noise_channels():
            if x.startswith("tphi"):
                tphi_supported_noise_channels.append(x)
        return tphi_supported_noise_channels
    

    def esys(self):
        if self.hamiltonian_creation == 'create_H':
            eigvals,eigvecs = torch.linalg.eigh(self.create_H())
        elif self.hamiltonian_creation == 'auto_H':
            eigvals,eigvecs = torch.linalg.eigh(self.auto_H())
        return eigvals,eigvecs
    
    def omega(self):
        eigvals,eigvecs = self.esys()
        ground_E = eigvals[0]
        excited_E = eigvals[1]
        return 2 * np.pi * (excited_E - ground_E) * 1e9

    #T1 CAPACITIVE FUNCTIONS

    def q_cap_fun(self) -> torch.Tensor:
        return (
            1e6
            * torch.pow((2 * np.pi * 6e9 / torch.abs(self.omega())) , 0.7)
        )

    def spectral_density_cap(self, T):
        
        therm_ratio = general.calc_therm_ratio(self.omega(), T)
        s = (
            2
            * 8
            * self.EC
            / self.q_cap_fun()
            * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
            / (1 + torch.exp(-therm_ratio))
        )
        s *= (
            2 * np.pi
        )  # We assume that system energies are given in units of frequency
        return s

#T1 CHARGE IMPEDANCE FUNCTIONS 

    def spectral_density_ci(self, R_0, T, R_k):
        # Note, our definition of Q_c is different from Zhang et al (2020) by a
        # factor of 2

        Q_c = R_k / (8 * np.pi * complex(R_0).real)
        therm_ratio = general.calc_therm_ratio(self.omega(), T)
        s = (
            2
            * ( self.omega() / 1e9 ) #annoying unit stuff - gotta convert back to GHz
            / Q_c
            * (1 / torch.tanh(0.5 * therm_ratio))
            / (1 + torch.exp(-therm_ratio))
        )
        return s

# T1 FLUX BIAS LINE FUNCTIONS

    def spectral_density_fbl(self,T,M, R_0):
        """
        Our definitions assume that the noise_op is dH/dflux.
        """

        therm_ratio = general.calc_therm_ratio(self.omega(), T)

        s = (
            2
            * (2 * np.pi) ** 2
            * M**2
            * ( self.omega() / 1e9 ) #annoying unit stuff - gotta convert back to GHz
            * sp.constants.hbar
            / complex(R_0).real
            * (1 / torch.tanh(0.5 * therm_ratio))
            / (1 + torch.exp(-therm_ratio))
        )
        # We assume that system energies are given in units of frequency and that
        # the noise operator to be used with this `spectral_density` is dH/dflux.
        # Hence we have to convert  2 powers of frequency to standard units
        s *= 1e9 ** 2.0
        return s
    #T1 INDUCTIVE FUNCTIONS

    def q_ind_fun(self, T):

        therm_ratio = abs(general.calc_therm_ratio(self.omega(), T))
        therm_ratio_500MHz = general.calc_therm_ratio(
            torch.tensor(2 * np.pi * 500e6), T)
        
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

    def spectral_density_ind(self,T):

        therm_ratio = abs(general.calc_therm_ratio(self.omega(), T))
        s = (
            2
            * self.EL
            / self.q_ind_fun(T)
            * (1 / torch.tanh(0.5 * torch.abs(therm_ratio)))
            / (1 + torch.exp(-therm_ratio))
        )
        s *= (
            2 * np.pi
        )  # We assume that system energies are given in units of frequency
        return s
    
    # T1 QUASIPARTICLE TUNNELLING FUNCTIONS

    def y_qp_fun(self, T, R_k):
        Delta = 3.4e-4
        x_qp = 3e-6
        omega = torch.abs(self.omega())
        Delta_in_Hz = Delta * sp.constants.e / sp.constants.h

        omega_in_Hz = omega / (2 * np.pi) 
        EJ_in_Hz = self.EJ * 1e9 # GHz to Hz

        therm_ratio = general.calc_therm_ratio(self.omega(), T)
        Delta_over_T = general.calc_therm_ratio(
            2 * np.pi * Delta_in_Hz, T
        )

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

    def spectral_density_qt(self, T, R_k):
        """Based on Eq. 19 in Smith et al (2020)."""
        therm_ratio = general.calc_therm_ratio(self.omega(), T)

        return (
            2
            * self.omega() / 1e9
            * complex(self.y_qp_fun(T, R_k)).real
            * (1 / torch.tanh(0.5 * therm_ratio))
            / (1 + torch.exp(-therm_ratio))
        )
    
    def qt_noise_op(self):
        argument = self.phi_operator() + 2 * np.pi * self.flux * torch.tensor(np.eye(self.dim),dtype=torch.double)
        qt_argument = argument/2
        return (torch.linalg.matrix_exp(qt_argument*1j)-torch.linalg.matrix_exp(qt_argument*-1j))/2j