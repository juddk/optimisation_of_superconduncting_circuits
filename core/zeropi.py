import scqubits as sc
import torch
import numpy as np
from scipy import sparse
from scipy.sparse import dia_matrix
import general
import scipy as sp
from typing import Union


class ZeroPi:
    def __init__(self, 
                 EJ: torch.Tensor, 
                 EL: torch.Tensor , 
                 ECJ: torch.Tensor, 
                 EC: torch.Tensor, 
                 dEJ: torch.Tensor, 
                 dCJ: torch.Tensor, 
                 flux: torch.Tensor, 
                 ng: float, 
                 ncut: float, 
                 truncated_dim: float, 
                 pt_count: int, 
                 min_val: float, 
                 max_val: float,
                 hamiltonian_creation: str):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ
        self.EC = EC
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.ng = ng  
        self.flux = flux  
        self.ncut = ncut  
        self.truncated_dim = truncated_dim   
        self.pt_count = pt_count 
        self.max_val = min_val  
        self.min_val = max_val
        self.hamiltonian_creation = hamiltonian_creation
    
    def t1_supported_noise_channels(self):
        t1_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid = sc.Grid1d(min_val= self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ   = self.EJ.item(),
            EL   = self.EL.item(), 
            ECJ  = self.ECJ.item(),
            EC   = self.EC.item(),
            ng   = self.ng,
            flux = self.flux.item(),
            ncut = self.ncut, 
            dEJ = self.dEJ, 
            dCJ = self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("t1"):
                t1_supported_noise_channels.append(x)
        return t1_supported_noise_channels

    def tphi_supported_noise_channels(self):
        tphi_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid = sc.Grid1d(min_val= self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ   = self.EJ.item(),
            EL   = self.EL.item(), 
            ECJ  = self.ECJ.item(),
            EC   = self.EC.item(),
            ng   = self.ng,
            flux = self.flux.item(),
            ncut = self.ncut, 
            dEJ = self.dEJ, 
            dCJ = self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("tphi"):
                tphi_supported_noise_channels.append(x)
        return tphi_supported_noise_channels

    def create_H():
        return

    def auto_H(self):

        create_qubit = sc.ZeroPi(
            grid = sc.Grid1d(min_val= self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ   = self.EJ.item(),
            EL   = self.EL.item(), 
            ECJ  = self.ECJ.item(),
            EC   = self.EC.item(),
            ng   = self.ng,
            flux = self.flux.item(),
            ncut = self.ncut, 
            dEJ = self.dEJ, 
            dCJ = self.dCJ,
        )
        
        return torch.from_numpy(create_qubit.hamiltonian().toarray())
    
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

    def _identity_theta(self) -> torch.Tensor:
        dim_theta = 2 * self.ncut + 1
        return torch.eye(dim_theta)
        

    def _cos_phi_operator(self, x)-> torch.Tensor:
        vals = np.cos(np.linspace(self.min_val, self.max_val, self.pt_count) + x)
        cos_phi_matrix = torch.from_numpy(dia_matrix((vals, [0]), shape=(self.pt_count, self.pt_count)).toarray())
        return cos_phi_matrix
        
    def _cos_theta_operator(self):
        dim_theta = 2 * self.ncut + 1
        cos_theta_matrix =  torch.from_numpy(
            (
            0.5
            * (
                sparse.dia_matrix(
                    ([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                )
                + sparse.dia_matrix(
                    ([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)
                )
            )
            ).toarray())
        return cos_theta_matrix
        

    def _sin_phi_operator(self,x)-> torch.Tensor:
        vals = np.sin(np.linspace(self.min_val, self.max_val, self.pt_count)+x)
        return torch.from_numpy(
            sparse.dia_matrix((vals, [0]), shape=(self.pt_count, self.pt_count)).toarray())


    def _sin_theta_operator(self)-> torch.Tensor:
        dim_theta = 2 * self.ncut + 1
        sin_theta_matrix = torch.from_numpy(
                -0.5
                * 1j
                * (
                    sparse.dia_matrix(
                        ([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                    )
                    - sparse.dia_matrix(
                        ([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)
                    )
                ).toarray()
                )
        return sin_theta_matrix
    

    def phi_operator(self)-> torch.Tensor:
        phi_operator = torch.kron(
            self._phi_operator(),
            self._identity_theta())
        return phi_operator
    #Noise.process_op(native_op=native, energy_esys=Noise.energy_esys(self.H()), truncated_dim=self.truncated_dim)
    


    def d_hamiltonian_d_EJ(self)-> torch.Tensor:
        d_potential_d_EJ_mat = -2.0 * torch.kron(
        self._cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
        self._cos_theta_operator()
        )
        return d_potential_d_EJ_mat
    #Noise.process_op(native_op=d_potential_d_EJ_mat, energy_esys=Noise().energy_esys, truncated_dim = self.truncated_dim)
    

    def d_hamiltonian_d_flux(self)-> torch.Tensor:
        op_1 = torch.kron(
                self._sin_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
                self._cos_theta_operator()
            )
        op_2 = torch.kron(
                self._cos_phi_operator(x=-2.0 * np.pi * self.flux / 2.0),
                self._sin_theta_operator()
            )
        d_potential_d_flux_mat =  -2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2
        
        return d_potential_d_flux_mat
    
    #process_op(native_op=d_potential_d_flux_mat, energy_esys=energy_esys, truncated_dim=truncated_dim)


    def _phi_operator(self)-> torch.Tensor:
        phi_matrix = sparse.dia_matrix((self.pt_count, self.pt_count))
        diag_elements = np.linspace(self.min_val, self.max_val, self.pt_count)
        phi_matrix.setdiag(diag_elements)

        return torch.from_numpy(phi_matrix.toarray())
    


    #technically the spectral densities should be the same cal for each qubit, however for ease we will define in each
    #qubit class
    
    def omega(self):
        eigvals,eigvecs = self.esys()
        ground = eigvecs[:,0]
        excited = eigvecs[:,1]
        ground_E = eigvals[0]
        excited_E = eigvals[1]
        return 2 * np.pi * (excited_E - ground_E) * 1e9

