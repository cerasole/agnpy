import numpy as np
from astropy import units as u
from astropy.constants import m_e, c
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.time_evolution import TimeEvolution, BlobExpansion, synchrotron_loss
import matplotlib.pyplot as plt


# set the quantities defining the blob and the electron distribution
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
W_e = 1e48 * u.erg # total energy in electrons

# initial electron distribution
n_e_initial = PowerLaw.from_total_energy(
    W_e,
    V_b,
    p=2.8,
    gamma_min=1e2,
    gamma_max=1e7,
    mass=m_e,
)

# define the blob and the energy loss mechanism
blob = Blob(R_b, n_e=n_e_initial)
synch = Synchrotron(blob)

# the blob radius grows at one tenth of the speed of light,
# and the magnetic field decays with the radius as B = B_0 * (R_0 / R)
expansion = BlobExpansion(v_exp=0.1 * c, magnetic_field_index=1.0)

# evolve over the time in which the radius doubles,
# considering synchrotron losses, adiabatic losses and density dilution
total_time = (R_b / expansion.v_exp).to("day")
time_evolution = TimeEvolution(blob, total_time, synchrotron_loss(synch), expansion=expansion)
time_evolution_result = time_evolution.evaluate()
print(f"Final blob radius: {blob.R_b:.2e}, final magnetic field: {blob.B:.3f}")

# let us plot both particle distributions, the initial and the evolved one
gamma = time_evolution_result.gamma
n_e_evol = time_evolution_result.density

fig, ax = plt.subplots()
n_e_initial.plot(ax=ax, gamma_power=2, label="initial distribution")
ax.plot(gamma, n_e_evol * gamma**2, label="evolved distribution (expanding blob)")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\gamma^2 n_e(\gamma)$")
ax.legend()
plt.show()
