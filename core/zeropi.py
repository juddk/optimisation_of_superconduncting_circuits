import scqubits as sc
import torch
import numpy as np
from scipy import sparse
from scipy.sparse import dia_matrix
import general
import scipy as sp
from typing import Union
from discretization import DOM
import xitorch
from xitorch import linalg

class ZeroPi:
    # All values in GHz
    def __init__(
        self,
        EJ: torch.Tensor,
        EL: torch.Tensor,
        ECJ: torch.Tensor,
        ECS: torch.Tensor,
        EC: torch.Tensor,
        dEJ: torch.Tensor,
        dCJ: torch.Tensor,
        phi_ext: torch.Tensor,
        varphi_ext: torch.Tensor,
        ng: float,
        ncut: float,
        truncated_dim: float,
        pt_count: int,
        min_val: float,
        max_val: float,
        hamiltonian_creation_solution #tuple[str, str] = ('manual_discretization','xitorch_davidon')
    ):
        self.EJ = EJ
        self.EL = EL
        self.ECJ = ECJ
        self.ECS= ECS
        self.EC = EC
        self.dEJ = dEJ
        self.dCJ = dCJ
        self.phi_ext = phi_ext
        self.varphi_ext = varphi_ext
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        self.pt_count = pt_count
        self.min_val = min_val
        self.max_val = max_val
        self.hamiltonian_creation_solution = hamiltonian_creation_solution

        """
        TILE

        Parameters
        ----------
        dddd :   diddd
        ddd :   flddd
        """

    # CREATING QUBIT


    def create_H(self):
        #this wont work gradients due to the sparse.dia_matrix
        return self.kinetic_matrix() + self.potential_matrix()

    def auto_H(self) -> torch.Tensor:
        create_qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.phi_ext.item(),
            ncut=self.ncut,
            dEJ=self.dEJ.item(),
            dCJ=self.dCJ.item(),
        )

        return torch.from_numpy(create_qubit.hamiltonian().toarray())
    
    def manual_discretization_H(self) -> torch.Tensor:
        # Constructs Hamiltonian using disretization of symbolic form.

        phi = np.linspace(0, 2 * np.pi, DOM.Nphi)
        cos_phi = np.cos(phi)
        cos_phi_m = np.diag(cos_phi)
        sin_phi_adj = np.sin(phi-self.phi_ext.detach().numpy()/2)
        sin_phi_adj_m = np.diag(sin_phi_adj)
        phi_m = np.diag(phi)
        _cos_phi = torch.kron(torch.tensor(cos_phi_m), torch.tensor(DOM.eye_Nphi) )
        _phi = torch.kron(torch.tensor(phi_m), torch.tensor(DOM.eye_Nphi) )
        _sin_phi_adj_m  = torch.kron(torch.tensor(sin_phi_adj_m), torch.tensor(DOM.eye_Nphi))

        theta = np.linspace(0, 2 * np.pi, DOM.Ntheta)
        cos_theta_adj = np.cos(theta-self.varphi_ext.detach().numpy()/2)
        sin_theta = np.sin(theta)
        cos_theta_adj_m = np.diag(cos_theta_adj)
        sin_theta_m = np.diag(sin_theta)
        _cos_theta_adj_m = torch.kron(torch.tensor(cos_theta_adj_m), torch.tensor(DOM.eye_Ntheta))
        _sin_theta_m = torch.kron(torch.tensor(sin_theta_m), torch.tensor(DOM.eye_Ntheta))


        I = torch.kron(torch.tensor(DOM.eye_Nphi), torch.tensor(DOM.eye_Ntheta))

        ham = -2 * self.ECJ * (DOM.partial_phi_fd * DOM.partial_phi_bk) \
            + 2 * self.ECS * (-1* DOM.partial_theta_fd**2 +self.ng**2*I-2*self.ng*DOM.partial_theta_fd)\
            + 2 * self.ECS * self.dCJ * DOM.partial_phi_fd * DOM.partial_theta_fd \
            - 2 * self.EJ * _cos_phi * _cos_theta_adj_m \
            + self.EL * _phi ** 2 \
            + 2 * self.EJ * I  \
            + self.EJ * self.dEJ * _sin_theta_m * _sin_phi_adj_m
        return ham + torch.transpose(ham, 1,0 )
       

    def t1_supported_noise_channels(self):
        t1_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.phi_ext.item(),
            ncut=self.ncut,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("t1"):
                t1_supported_noise_channels.append(x)
        return t1_supported_noise_channels

    def tphi_supported_noise_channels(self):
        tphi_supported_noise_channels = []
        qubit = sc.ZeroPi(
            grid=sc.Grid1d(min_val=self.min_val, max_val=self.max_val, pt_count=self.pt_count),
            EJ=self.EJ.item(),
            EL=self.EL.item(),
            ECJ=self.ECJ.item(),
            EC=self.EC.item(),
            ng=self.ng,
            flux=self.phi_ext.item(),
            ncut=self.ncut,
            dEJ=self.dEJ,
            dCJ=self.dCJ,
        )
        for x in qubit.supported_noise_channels():
            if x.startswith("tphi"):
                tphi_supported_noise_channels.append(x)
        return tphi_supported_noise_channels

    def esys(self):
        if self.hamiltonian_creation_solution == "create_H":
            eigvals, eigvecs = torch.linalg.eigh(self.create_H())
        elif self.hamiltonian_creation_solution == "auto_H":
            eigvals, eigvecs = torch.linalg.eigh(self.auto_H())
        elif self.hamiltonian_creation_solution == "manual_discretization_davidson":
            H = xitorch.LinearOperator.m(self.manual_discretization_H())
            xitorch.LinearOperator._getparamnames(H, "EJ, EJ, EC")
            eigvals, eigvecs = xitorch.linalg.symeig(H, 2, method="davidson", max_niter=100, nguess=None, v_init="randn", max_addition=None, min_eps=1e-06, verbose=False)
        return eigvals, eigvecs

    # OPERATORS
    def _identity_theta(self) -> torch.Tensor:
        dim_theta = 2 * self.ncut + 1
        return torch.eye(dim_theta)

    ###we may need x to be a tensor - see uses of the _cos_phi_operator and _sin_phi_operator
    def _cos_phi_operator(self, x) -> torch.Tensor:
        vals = np.cos(np.linspace(self.min_val, self.max_val, self.pt_count) + x)
        cos_phi_matrix = torch.from_numpy(dia_matrix((vals, [0]), shape=(self.pt_count, self.pt_count)).toarray())
        return cos_phi_matrix

    def _cos_theta_operator(self):
        dim_theta = 2 * self.ncut + 1
        cos_theta_matrix = torch.from_numpy(
            (
                0.5
                * (
                    sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))
                    + sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta))
                )
            ).toarray()
        )
        return cos_theta_matrix

    def _sin_phi_operator(self, x) -> torch.Tensor:
        vals = np.sin(np.linspace(self.min_val, self.max_val, self.pt_count) + x)
        return torch.from_numpy(sparse.dia_matrix((vals, [0]), shape=(self.pt_count, self.pt_count)).toarray())

    def _sin_theta_operator(self) -> torch.Tensor:
        dim_theta = 2 * self.ncut + 1
        sin_theta_matrix = torch.from_numpy(
            -0.5
            * 1j
            * (
                sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))
                - sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta))
            ).toarray()
        )
        return sin_theta_matrix

    def phi_operator(self) -> torch.Tensor:
        phi_operator = torch.kron(self._phi_operator(), self._identity_theta())

        return phi_operator

    def _phi_operator(self) -> torch.Tensor:
        phi_matrix = sparse.dia_matrix((self.pt_count, self.pt_count))
        diag_elements = np.linspace(self.min_val, self.max_val, self.pt_count)
        phi_matrix.setdiag(diag_elements)

        return torch.from_numpy(phi_matrix.toarray())

    def d_hamiltonian_d_EJ(self) -> torch.Tensor:
        d_potential_d_EJ_mat = -2.0 * torch.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.phi_ext.item() / 2.0),
            self._cos_theta_operator(),
        )

        ###the flux.item() could be an issue for computing gradients
        return d_potential_d_EJ_mat

    def d_hamiltonian_d_flux(self) -> torch.Tensor:
        op_1 = torch.kron(
            self._sin_phi_operator(x=-2.0 * np.pi * self.phi_ext.item() / 2.0),
            self._cos_theta_operator(),
        )
        op_2 = torch.kron(
            self._cos_phi_operator(x=-2.0 * np.pi * self.phi_ext.item() / 2.0),
            self._sin_theta_operator(),
        )
        d_potential_d_flux_mat = -2.0 * np.pi * self.EJ * op_1 - np.pi * self.EJ * self.dEJ * op_2

        ###the flux.item() could be an issue for computing gradients
        return d_potential_d_flux_mat

    # OPERATORS NEEDED FOR HAMILTONIAN CREATION

    def band_matrix(band_coeffs, band_offsets, dim, dtype, has_corners):
        ones_vector = np.ones(dim)
        vectors = [ones_vector * number for number in band_coeffs]
        matrix = sparse.dia_matrix((vectors, band_offsets), shape=(dim, dim), dtype=dtype)
        if not has_corners:
            return matrix.tocsc()
        for index, offset in enumerate(band_offsets):
            if offset < 0:
                corner_offset = dim + offset
                corner_band = vectors[index]
                corner_band = corner_band[offset:]
            elif offset > 0:
                corner_offset = -dim + offset
                corner_band = vectors[index][:-offset]
                corner_band = corner_band[-offset:]
            else:  # when offset == 0
                continue
            matrix.setdiag(corner_band, k=corner_offset)
        return torch.from_numpy(matrix.toarray())

    def first_derivative_matrix(self, prefactor):
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_

        delta_x = (self.max_val - self.min_val) / (self.pt_count - 1)
        matrix_diagonals = [
            coefficient * prefactor / delta_x for coefficient in [-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60]
        ]
        offset = [i - (7 - 1) // 2 for i in range(7)]
        print(matrix_diagonals)
        print(offset)
        print(self.pt_count)
        print(dtp)

        return self.band_matrix(matrix_diagonals, offset, self.pt_count, dtp, False)

    def second_derivative_matrix(self, prefactor):
        if isinstance(prefactor, complex):
            dtp = np.complex_
        else:
            dtp = np.float_

        delta_x = (self.max_val - self.min_val) / (self.pt_count - 1)
        matrix_diagonals = [
            coefficient * prefactor / delta_x**2
            for coefficient in [
                1 / 90,
                -3 / 20,
                3 / 2,
                -49 / 18,
                3 / 2,
                -3 / 20,
                1 / 90,
            ]
        ]
        offset = [i - (7 - 1) // 2 for i in range(7)]
        return self.band_matrix(matrix_diagonals, offset, self.pt_count, dtp, False)

    def i_d_dphi_operator(self):
        return torch.kron(self.first_derivative_matrix(prefactor=1j), self._identity_theta())

    def n_theta_operator(self):
        dim_theta = 2 * self.ncut + 1
        diag_elements = np.arange(-self.ncut, self.ncut + 1)
        n_theta_matrix = torch.from_numpy(
            sparse.dia_matrix((diag_elements, [0]), shape=(dim_theta, dim_theta)).tocsc().toarray()
        )
        return torch.kron(self._identity_phi(), n_theta_matrix)

    def _identity_phi(self):
        return torch.eye(self.pt_count)

    def sin_theta_operator(self):
        return torch.kron(self._identity_phi(), self._sin_theta_operator())

    # Kinetic Part
    def kinetic_matrix(self):
        dim_theta = 2 * self.ncut + 1
        identity_phi = torch.eye(self.pt_count)
        identity_theta = torch.eye(dim_theta)
        kinetic_matrix_phi = self.second_derivative_matrix(prefactor=-2.0 * self.ECJ)

        diag_elements = 2.0 * self.ECS * torch.square(torch.arange(-self.ncut + self.ng, self.ncut + 1 + self.ng))

        ##THIS WONT WORK FOR COMPUTING GRAD
        kinetic_matrix_theta = torch.from_numpy(
            sparse.dia_matrix((diag_elements, [0]), shape=(dim_theta, dim_theta)).tocsc().toarray()
        )

        kinetic_matrix = torch.kron(kinetic_matrix_phi, identity_theta) + torch.kron(identity_phi, kinetic_matrix_theta)
        if self.dCJ != 0:
            kinetic_matrix -= 2.0 * self.ECS * self.dCJ * self.i_d_dphi_operator() * self.n_theta_operator()
        return kinetic_matrix

    # Potential Part
    def potential_matrix(self):
        grid_linspace = torch.linspace(self.min_value, self.max_value, self.pt_count)
        dim_theta = 2 * self.ncut + 1

        phi_inductive_vals = self.EL * torch.square(grid_linspace)

        phi_inductive_potential = torch.from_numpy(
            sparse.dia_matrix((phi_inductive_vals, [0]), shape=(self.pt_count, self.pt_count)).tocsc().toarray()
        )

        phi_cos_vals = torch.cos(grid_linspace - 2.0 * np.pi * self.phi_ext / 2.0)

        phi_cos_potential = torch.from_numpy(
            sparse.dia_matrix((phi_cos_vals, [0]), shape=(self.pt_count, self.pt_count)).tocsc().toarray()
        )

        phi_sin_vals = torch.sin(grid_linspace - 2.0 * np.pi * self.phi_ext / 2.0)

        phi_sin_potential = torch.from_numpy(
            sparse.dia_matrix((phi_sin_vals, [0]), shape=(self.pt_count, self.pt_count)).tocsc().toarray()
        )

        theta_cos_potential = torch.from_numpy(
            (
                -self.EJ
                * (
                    sparse.dia_matrix(([1.0] * dim_theta, [-1]), shape=(dim_theta, dim_theta))
                    + sparse.dia_matrix(([1.0] * dim_theta, [1]), shape=(dim_theta, dim_theta))
                )
            )
            .tocsc()
            .toarray()
        )

        potential_mat = (
            torch.kron(phi_cos_potential, theta_cos_potential)
            + torch.kron(phi_inductive_potential, self._identity_theta())
            + 2 * self.EJ * torch.kron(self._identity_phi(), self._identity_theta())
        )
        if self.dEJ != 0:
            potential_mat += (
                self.EJ * self.dEJ * torch.kron(phi_sin_potential, self._identity_theta()) * self.sin_theta_operator()
            )
        return potential_mat
