#discretization


import torch 
import scipy as sp
from scipy import sparse as sps

#currently set up 2D - somehow need to genelise this to work for an aribtrary chosen dimension 

class DOM: 

    def __init__(self, 
                Nphi = 100,
                Ntheta = 100, 
                ):
        """
        Differential Operator Matrices

        Parameters
        ----------
        Nx, Ny  :   discretization dimension for each  
        dx, dy  :   float
        """

    Nphi = 100
    Ntheta = 100
    eye_Nphi = torch.eye(Nphi)
    eye_Ntheta = torch.eye(Ntheta)
    partial_phi_fd = torch.kron(eye_Ntheta, torch.tensor(sps.diags([-1, 1, 1], [0, 1, -Nphi+1], shape=(Nphi, Nphi)).todense()))
    partial_phi_bk = torch.kron(eye_Ntheta, torch.tensor(sps.diags([1, -1, -1], [0, -1, Nphi-1], shape=(Nphi, Nphi)).todense()))
    partial_theta_fd = torch.kron(eye_Nphi, torch.tensor(sps.diags([-1, 1, 1], [0, 1, -Ntheta+1], shape=(Ntheta, Ntheta)).todense()))
    partial_theta_bk =torch.kron(eye_Nphi, torch.tensor(sps.diags([1, -1, -1], [0, -1, Ntheta-1], shape=(Ntheta, Ntheta)).todense()))
