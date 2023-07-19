import torch
import general
import scqubits as sc
import numpy as np
import math
import scipy as sp


class Fluxonium:
    def __init__(
        self,
        EJ: torch.Tensor,
        EC: torch.Tensor,
        EL: torch.Tensor,
        flux: torch.Tensor,
        dim: int,
        hamiltonian_creation: str,
    ):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.flux = flux
        self.dim = dim
        self.hamiltonian_creation = hamiltonian_creation

    # CREATING QUBIT
    def create_H(self) -> torch.Tensor:
        # Constructs Hamiltonian matrix in harmonic-oscillator basis.
        # Following Zhu et al., PRB 87, 024510 (2013)
        # Consutruction taken from scqubits source code

        plasma_energy = torch.sqrt(8.0 * self.EL * self.EC)
        diag_elements = [(i + 0.5) for i in range(self.dim)]
        lc_osc = torch.tensor(np.diag(diag_elements), dtype=torch.double)
        lc_osc = lc_osc * plasma_energy
        cos_phi = self.cos_phi_operator(beta=2 * np.pi * self.flux)

        return lc_osc - self.EJ * cos_phi

    def auto_H(self) -> torch.Tensor:
        # Constructs Hamiltonian in harmonic-oscillator basis using scqubits.
        return torch.from_numpy(
            sc.Fluxonium(
                EJ=self.EJ.item(),
                EC=self.EC.item(),
                EL=self.EL.item(),
                flux=self.flux.item(),
                cutoff=self.dim,
            ).hamiltonian()
        )

    def t1_supported_noise_channels(self):
        t1_supported_noise_channels = []
        for x in sc.Fluxonium(
            EJ=self.EJ.item(),
            EC=self.EC.item(),
            EL=self.EL.item(),
            flux=self.flux.item(),
            cutoff=self.dim,
        ).supported_noise_channels():
            if x.startswith("t1"):
                t1_supported_noise_channels.append(x)
        return t1_supported_noise_channels

    def tphi_supported_noise_channels(self):
        tphi_supported_noise_channels = []
        for x in sc.Fluxonium(
            EJ=self.EJ.item(),
            EC=self.EC.item(),
            EL=self.EL.item(),
            flux=self.flux.item(),
            cutoff=self.dim,
        ).supported_noise_channels():
            if x.startswith("tphi"):
                tphi_supported_noise_channels.append(x)
        return tphi_supported_noise_channels

    def esys(self):
        # Hamiltonian given in harmonic oscillator basis
        # TBC: eigenvals are returned in units of freqeucy
        if self.hamiltonian_creation == "create_H":
            eigvals, eigvecs = torch.linalg.eigh(self.create_H())
        elif self.hamiltonian_creation == "auto_H":
            eigvals, eigvecs = torch.linalg.eigh(self.auto_H())
        return eigvals, eigvecs

    # OPERATORS
    def phi_operator(self) -> torch.Tensor:
        # Returns the phi operator in the LC harmonic oscillator basis

        phi = (
            (
                torch.tensor(general.creation(self.dim), dtype=torch.double)
                + torch.tensor(general.annihilation(self.dim), dtype=torch.double)
            )
            * self.phi_osc()
            / math.sqrt(2)
        )

        return phi

    def cos_phi_operator(self, alpha: float = 1.0, beta: float = 0.0):
        # Returns the cos phi operator in the LC harmonic oscillator basis

        argument = alpha * self.phi_operator() + beta * torch.eye(self.dim)

        # since pytorch does not have a cosm fucntion, use exponential form
        cos_phi = (
            torch.linalg.matrix_exp(argument * 1j)
            + torch.linalg.matrix_exp(argument * -1j)
        ) / 2

        return cos_phi

    def sin_phi_operator(self, alpha: float = 1.0, beta: float = 0.0):
        # Returns the sin phi operator in the LC harmonic oscillator basis
        argument = alpha * self.phi_operator() + beta * torch.eye(self.dim)
        sin_phi = (
            torch.linalg.matrix_exp(argument * 1j)
            - torch.linalg.matrix_exp(argument * -1j)
        ) / 2j

        return sin_phi

    def n_operator(self):
        # Returns the n operator in the LC harmonic oscillator basis
        n_op = (
            1j
            * (
                torch.tensor(general.creation(self.dim), dtype=torch.double)
                - torch.tensor(general.annihilation(self.dim), dtype=torch.double)
            )
            / (self.phi_osc() * math.sqrt(2))
        )
        return n_op

    def phi_osc(self):
        # LC oscillator length
        return torch.pow((8.0 * self.EC / self.EL), 0.25)

    def d_hamiltonian_d_flux(self):
        # Returns operator representing a derivative of the Hamiltonian with respect to
        # flux in the harmonic-oscillator (charge) basis

        d_ham_d_flux = (
            -2
            * np.pi
            * self.EJ
            * self.sin_phi_operator(alpha=1, beta=2 * np.pi * self.flux)
        )
        return d_ham_d_flux

    def d_hamiltonian_d_EJ(self):
        # Returns operator representing a derivative of the Hamiltonian with respect to
        # EJ in the harmonic-oscillator (charge) basis
        return -self.cos_phi_operator(alpha=1, beta=2 * np.pi * self.flux)
