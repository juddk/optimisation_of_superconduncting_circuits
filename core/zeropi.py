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

    #CREATING QUBIT
    def create_H():
        def first_derivative_matrix(prefactor):
            #NEED TO FILL IN
            return

        def second_derivative_matrix(prefactor):
            #NEED TO FILL IN
            return

        #Kinetic Part
        def kinetic_matrix():
            identity_phi = torch.eye(pt_count)
            identity_theta = torch.eye(dim_theta)
            i_d_dphi_operator = torch.kron(first_derivative_matrix(prefactor=1j), identity_theta )
            kinetic_matrix_phi = second_derivative_matrix(prefactor =  -2.0*ECJ)
            ##
            diag_elements = 2.0 * ECS * torch.square(torch.arange(-ncut + ng, ncut + 1 + ng))
            kinetic_matrix_theta = torch.diag_embed(diag_elements, [0])  ### Need to figure out what is spare.dia_matrix is doing
            #
            diag_elements = torch.arange(-ncut, ncut + 1)
            n_theta_matrix= torch.diag_embed(diag_elements, [0])  ### Need to figure out what is spare.dia_matrix is doing
            #### native = sparse.kron(self._identity_phi(), n_theta_matrix, format="csc")
            ##### n_theta_operator = process_op(native_op=native, energy_esys=energy_esys)
            #
            kinetic_matrix = torch.kron(kinetic_matrix_phi, identity_theta)+ torch.kron(identity_phi, kinetic_matrix_theta)
            if dCJ != 0:
                kinetic_matrix -= (
                            2.0
                            * ECS
                            * dCJ
                            * i_d_dphi_operator()
                            * n_theta_operator()
                )
            return kinetic_matrix
        #Potential Part 
        def potential_matrix():
            grid_linspace = self.grid.make_linspace()
            phi_inductive_vals = self.EL * np.square(grid_linspace)
            phi_inductive_potential = sparse.dia_matrix(
                    (phi_inductive_vals, [0]), shape=(pt_count, pt_count)
                ).tocsc()
            phi_cos_vals = np.cos(grid_linspace - 2.0 * np.pi * self.flux / 2.0)
            phi_cos_potential = sparse.dia_matrix(
                    (phi_cos_vals, [0]), shape=(pt_count, pt_count)
                ).tocsc()
            phi_sin_vals = np.sin(grid_linspace - 2.0 * np.pi * self.flux / 2.0)
            phi_sin_potential = sparse.dia_matrix(
                    (phi_sin_vals, [0]), shape=(pt_count, pt_count)
                ).tocsc()
            theta_cos_potential = (
                    -self.EJ
                    * (
                        sparse.dia_matrix(
                            ([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta)
                        )
                        + sparse.dia_matrix(
                            ([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta)
                        )
                    )
                ).tocsc()
            potential_mat = (
                    sparse.kron(phi_cos_potential, theta_cos_potential, format="csc")
                    + sparse.kron(phi_inductive_potential, self._identity_theta(), format="csc")
                    + 2
                    * self.EJ
                    * sparse.kron(self._identity_phi(), self._identity_theta(), format="csc")
                )     
            if dEJ != 0:
                potential_mat += (
                        EJ
                        * dEJ
                        * sparse.kron(phi_sin_potential, self._identity_theta(), format="csc")
                        * self.sin_theta_operator()
                    )
            return potential_mat
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
            dEJ = self.dEJ.item(), 
            dCJ = self.dCJ.item(),
        )
        
        native_basis = torch.from_numpy(create_qubit.hamiltonian().toarray())
        
        return native_basis
    
    
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

   
    def esys(self):
        if self.hamiltonian_creation == 'create_H':
            eigvals,eigvecs = torch.linalg.eigh(self.create_H())
        elif self.hamiltonian_creation == 'auto_H':
            eigvals,eigvecs = torch.linalg.eigh(self.auto_H())
        return eigvals,eigvecs
    
    def omega(self):
        #omega in units of radian per second
        ### in zero pi do we take the 1st and 2nd Energy levels?
        eigvals = self.esys()[0]
        ground_E = eigvals[0]
        excited_E = eigvals[1]
        return 2 * np.pi * (excited_E - ground_E) 
    

    #OPERATORS
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
        eigvecs = self.esys()[1]
        return general.change_operator_basis(phi_operator, eigvecs)
  

    def _phi_operator(self)-> torch.Tensor:
        phi_matrix = sparse.dia_matrix((self.pt_count, self.pt_count))
        diag_elements = np.linspace(self.min_val, self.max_val, self.pt_count)
        phi_matrix.setdiag(diag_elements)

        return torch.from_numpy(phi_matrix.toarray())
    

    def d_hamiltonian_d_EJ(self)-> torch.Tensor:
        d_potential_d_EJ_mat = -2.0 * torch.kron(
        self._cos_phi_operator(x=-2.0 * np.pi * self.flux.item() / 2.0),
        self._cos_theta_operator()
        )
        eigvecs = self.esys()[1]
        return general.change_operator_basis(d_potential_d_EJ_mat, eigvecs)
        
   

    def d_hamiltonian_d_flux(self)-> torch.Tensor:
        op_1 = torch.kron(
                self._sin_phi_operator(x=-2.0 * np.pi * self.flux.item() / 2.0),
                self._cos_theta_operator()
            )
        op_2 = torch.kron(
                self._cos_phi_operator(x=-2.0 * np.pi * self.flux.item() / 2.0),
                self._sin_theta_operator()
            )
        d_potential_d_flux_mat =  -2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2
    
        eigvecs = self.esys()[1]
        return general.change_operator_basis(d_potential_d_flux_mat, eigvecs)

    
   