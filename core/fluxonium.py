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

    #CREATING QUBIT
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
        #omega in units of radian per second
        eigvals = self.esys()[0]
        ground_E = eigvals[0]
        excited_E = eigvals[1]
        return 2 * np.pi * (excited_E - ground_E) 

    #OPERATORS
    def phi_operator(self):
          phi_osc = torch.pow((8.0 * self.EC/self.EL) , 0.25)
          phi = ((torch.tensor(general.creation(self.dim),dtype=torch.double) 
                + torch.tensor(general.annihilation(self.dim),dtype=torch.double))
                * phi_osc/ math.sqrt(2))
          #in scqubits they convert to energy eigenbasis but causes circular problem here as this is used to define H
          return phi

    def cos_phi_operator(self):
        argument = self.phi_operator() + 2 * np.pi * self.flux * torch.tensor(np.eye(self.dim),dtype=torch.double)
        cos_phi = (torch.linalg.matrix_exp(argument*1j)+torch.linalg.matrix_exp(argument*-1j))/2
        #in scqubits they convert to energy eigenbasis but causes circular problem here as this is used to define H
        return cos_phi
    
    def sin_phi_operator(self):
        argument = self.phi_operator() + 2 * np.pi * self.flux * torch.tensor(np.eye(self.dim),dtype=torch.double)
        sin_phi = (torch.linalg.matrix_exp(argument*1j)-torch.linalg.matrix_exp(argument*-1j))/2j
        return general.change_operator_basis(sin_phi, self.esys()[1])
    
    def n_operator(self):
        phi_osc = torch.pow((8.0 * self.EC/ self.EL) , 0.25)
        n_op = (1j
                * (torch.tensor(general.creation(self.dim),dtype=torch.double) - torch.tensor(general.annihilation(self.dim),dtype=torch.double))
                / (phi_osc * math.sqrt(2)))
        return general.change_operator_basis(n_op, self.esys()[1])
    
    def d_hamiltonian_d_flux(self):
        d_ham_d_flux = -2 * np.pi * self.EJ * self.sin_phi_operator()
        return general.change_operator_basis(d_ham_d_flux, self.esys()[1])

    def d_hamiltonian_d_EJ(self):
        return general.change_operator_basis(-self.cos_phi_operator(), self.esys()[1])
    