import numpy as np
import astropy.units as u
from astropy.constants import m_e
from astropy.coordinates import Distance
import matplotlib.pyplot as plt

from agnpy.emission_regions import Blob
from agnpy.spectra import PowerLaw
from agnpy.synchrotron import Synchrotron
from agnpy.utils.plot import plot_sed, load_mpl_rc

load_mpl_rc()


def polarization_degree_power_law(p):
    return (p + 1) / (p + 7 / 3)


# set the quantities defining the blob
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b**3
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
B = 1 * u.G

# electron distribution
W_e = 1e46 * u.erg  # total energy in electrons
p = 2.0

n_e = PowerLaw.from_total_energy(
    W_e,
    V_b,
    p=p,
    gamma_min=1e2,
    gamma_max=1e7,
    mass=m_e,
)

# define the emission region and the radiative process
blob = Blob(R_b, z, delta_D, Gamma, B, n_e=n_e)
synch = Synchrotron(blob=blob)

# compute the syncrotron SED and polarization degree over an array of frequencies
nu = np.logspace(15, 24) * u.Hz
synch_sed = synch.sed_flux(nu)
pol_degree = synch.polarization_degree(nu)

# plot
fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

plot_sed(nu, synch_sed, ax=axs[0], label="Blob, synchrotron", lw=3)
m = synch_sed.max().to_value("erg cm-2 s-1")
axs[0].set_ylim(top=m * 1.5, bottom=m / 1e6)

ax = axs[1]
ax.plot(nu, pol_degree, label="Blob, synchrotron", lw=3)
ax.axhline(
    polarization_degree_power_law(p),
    ls="dashed",
    lw=2,
    color="tab:orange",
    label=f"Power-law with index $p={p:.1f}$",
)
ax.set_xlabel(axs[0].get_xlabel())
ax.set_ylabel("$\\Pi_{\\nu}$")
ax.axhline(1.0, ls="dotted", lw=2, color="black", label="Limit at high frequencies")
ax.legend()

plt.show()
