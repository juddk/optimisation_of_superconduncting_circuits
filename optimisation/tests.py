# Unit Tests
import torch
import numpy as np
import scipy as sp
import sys

sys.path.append("/Users/judd/Documents/optimisation_of_superconduncting_circuits/core")
from fluxonium import Fluxonium
from zeropi import ZeroPi
import general as general
import scqubits as sc

# Fluxonium Class T2 Calc

EJ = torch.rand(1, requires_grad=True, dtype=torch.double)
EC = torch.rand(1, requires_grad=True, dtype=torch.double)
EL = torch.rand(1, requires_grad=True, dtype=torch.double)
flux = torch.tensor([0.5], requires_grad=True, dtype=torch.double)

EJ.data = EJ.data * (20 - 2.5) + 2.5
EC.data = EC.data * (8 - 1e-3) + 1e-3
EL.data = EL.data * (10 - 2e-1) + 2e-1

dim = 110

fluxonium_auto = Fluxonium(EJ, EC, EL, flux, dim, "auto_H")
fluxonium_create = Fluxonium(EJ, EC, EL, flux, dim, "create_H")
fluxonium_sym = Fluxonium(EJ, EC, EL, flux, dim, "sym_H")
fluxonium_sc = sc.Fluxonium(EJ=EJ.item(), EC=EC.item(), EL=EL.item(), flux=flux.item(), cutoff=dim)

general.t2_rate(
    fluxonium_create, fluxonium_create.t1_supported_noise_channels(), fluxonium_create.tphi_supported_noise_channels()
)
print(
    f"Fluxonium Class: t1_quasiparticle_tunneling = {(fluxonium_sc.t1_quasiparticle_tunneling()*general.effective_t1_rate(fluxonium_create, 't1_quasiparticle_tunneling').item())}"
)
print(
    f"Fluxonium Class: t1_charge_impedance = {(fluxonium_sc.t1_charge_impedance()*general.effective_t1_rate(fluxonium_create , 't1_charge_impedance').item())}"
)
print(
    f"Fluxonium Class: t1_capacitive = {(fluxonium_sc.t1_capacitive()*general.effective_t1_rate(fluxonium_create, 't1_capacitive').item())}"
)
print(
    f"Fluxonium Class: t1_inductive = {(fluxonium_sc.t1_inductive() * general.effective_t1_rate(fluxonium_create, 't1_inductive').item())}"
)
print(
    f"Fluxonium Class: t1_flux_bias_line = {(fluxonium_sc.t1_flux_bias_line() * general.effective_t1_rate(fluxonium_create , 't1_flux_bias_line').item())}"
)
print(
    f"Fluxonium Class: tphi_1_over_f_cc = {(fluxonium_sc.tphi_1_over_f_cc() * general.effective_tphi_rate(fluxonium_create , 'tphi_1_over_f_cc').item())}"
)
print(
    f"Fluxonium Class: tphi_1_over_f_cc = {(fluxonium_sc.tphi_1_over_f_flux() * general.effective_tphi_rate(fluxonium_create, 'tphi_1_over_f_flux').item())}"
)


# Zeropi Class T2 Calc
EJ = torch.tensor(10.00, requires_grad=True, dtype=torch.double)
EL = torch.tensor(0.04, requires_grad=True, dtype=torch.double)
ECJ = torch.tensor(20, requires_grad=True, dtype=torch.double)
EC = torch.tensor(0.04, requires_grad=True, dtype=torch.double)
dEJ = torch.tensor(0.0, requires_grad=True, dtype=torch.double)
dCJ = torch.tensor(0.0, requires_grad=True, dtype=torch.double)
flux = torch.tensor(0.23, requires_grad=True, dtype=torch.double)
ng = 0.1
ncut = 30
truncated_dim = 10
pt_count = 10
min_val = -19
max_val = 19
hamiltonian_creation = "auto_H"

zeropi = ZeroPi(
    EJ=EJ,
    EL=EL,
    ECJ=ECJ,
    EC=EC,
    dEJ=dEJ,
    dCJ=dCJ,
    flux=flux,
    ng=ng,
    ncut=ncut,
    truncated_dim=truncated_dim,
    pt_count=pt_count,
    min_val=min_val,
    max_val=max_val,
    hamiltonian_creation=hamiltonian_creation,
)

zeropi_sc = sc.ZeroPi(
    grid=sc.Grid1d(min_val=min_val, max_val=max_val, pt_count=pt_count),
    EJ=EJ.item(),
    EL=EL.item(),
    ECJ=ECJ.item(),
    EC=EC.item(),
    ng=ng,
    flux=flux.item(),
    ncut=ncut,
    dEJ=dEJ.item(),
    dCJ=dCJ.item(),
)

print(zeropi_sc.t1_inductive() * general.effective_t1_rate(zeropi, "t1_inductive").item())
print(zeropi_sc.t1_flux_bias_line() * general.effective_t1_rate(zeropi, "t1_flux_bias_line").item())
print(zeropi_sc.tphi_1_over_f_cc() * general.effective_tphi_rate(zeropi, "tphi_1_over_f_cc").item())
print(zeropi_sc.tphi_1_over_f_flux() * general.effective_tphi_rate(zeropi, "tphi_1_over_f_flux").item())

print("____")

print(zeropi_sc.t1_flux_bias_line())
print(1 / general.effective_t1_rate(zeropi, "t1_flux_bias_line").item())


# Arbitrary Circuits Fluxonium T2 Calc


# Arbitrary Circuits Zeropi T2 Calc
