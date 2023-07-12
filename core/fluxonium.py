import torch
import noise
import scqubits as sc
import numpy as np
import math


class Fluxonium:
    def __init__(self, EJ,EC,EL,flux,dim):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux = flux 
        self.dim = dim


    def create_H(self):

        plasma_energy = torch.sqrt(8.0*torch.tensor(self.EL) *torch.tensor(self.EC))
        diag_elements = [(i + 0.5) for i in range(self.dim)]
        lc_osc = torch.tensor(np.diag(diag_elements),dtype=torch.double)
        lc_osc = lc_osc * plasma_energy

        phi_osc = torch.pow((8.0 * torch.tensor(self.EC)/ torch.tensor(self.EL)) , 0.25)
        phi = (
                (torch.tensor(noise.creation(self.dim),dtype=torch.double) + torch.tensor(noise.annihilation(self.dim),dtype=torch.double))
                * phi_osc/ math.sqrt(2)
            )

        argument = phi + 2 * np.pi * self.flux * torch.tensor(np.eye(self.dim),dtype=torch.double)

        cos_phi = (torch.linalg.matrix_exp(argument*1j)+torch.linalg.matrix_exp(argument*-1j))/2
        sin_phi = (torch.linalg.matrix_exp(argument*1j)-torch.linalg.matrix_exp(argument*-1j))/2j

        n_op = (
                    1j
                    * (torch.tensor(noise.creation(self.dim),dtype=torch.double) - torch.tensor(noise.annihilation(self.dim),dtype=torch.double))
                    / (phi_osc * math.sqrt(2))
                )

        d_ham_d_flux = -2 * np.pi * torch.tensor(self.EJ) * sin_phi
        d_ham_d_EJ = - cos_phi

        return lc_osc - self.EJ*cos_phi

    
    def auto_H(self):
        return torch.from_numpy(
            sc.Fluxonium(EJ=self.EJ, EC=self.EC, EL = self.EL, flux = self.flux, cutoff= self.dim).hamiltonian())


    #T1 CAPACITIVE FUNCTIONS