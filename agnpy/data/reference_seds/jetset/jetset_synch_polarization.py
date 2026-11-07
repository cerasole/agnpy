# generate a synchrotron SED and polarization degree to be confronted with agnpy
import numpy as np
import astropy.units as u
from astropy.constants import m_e
from astropy.table import QTable
import matplotlib.pyplot as plt
from jetset.jet_model import Jet
from agnpy.emission_regions import Blob
from agnpy.spectra import PowerLaw


def polarization_degree_power_law(p):
    return (p + 1) / (p + 7 / 3)


tab = QTable()

# agnpy
# - blob
spectral_indices = [1.50, 2.00, 2.50, 3.00]
R_b = 2.5e16 * u.cm
B = 6.5e-2 * u.G
Gamma = 15.0
theta = 2.7 / 2 * u.Unit("deg")
Beta = np.sqrt(1 - 1 / Gamma**2)
delta_D = 1 / (Gamma * (1 - Beta * np.cos(theta.to_value("rad"))))
z = 0.2
N = 1.2e0 * u.Unit("cm-3")
gamma_min = 50
gamma_max = 8e5

#
for i, p in enumerate(spectral_indices):
    n_e = PowerLaw.from_total_density(
        N, p=p, gamma_min=gamma_min, gamma_max=gamma_max, mass=m_e
    )
    blob = Blob(n_e=n_e, R_b=R_b, z=z, B=B, delta_D=delta_D, Gamma=Gamma)

    # jet(set) with power-law electron distribution
    jet = Jet(
        name="pl",
        electron_distribution="pl",
        electron_distribution_log_values=False,
        beaming_expr="bulk_theta",
    )
    # - blob
    jet.set_par("N", val=blob.n_e_tot.value)
    jet.set_par("p", val=blob.n_e.p)
    jet.set_par("gmin", val=blob.n_e.gamma_min)
    jet.set_par("gmax", val=blob.n_e.gamma_max)
    jet.set_par("R", val=blob.R_b.value)
    jet.set_par("B", val=blob.B.value)
    jet.set_par("BulkFactor", val=blob.Gamma)
    jet.set_par("theta", val=theta.to_value("deg"))
    jet.set_par("z_cosm", val=blob.z)
    jet.electron_distribution.update()
    jet.set_nu_grid(1e8, 1e20, 300)
    # remove SSA
    jet.spectral_components.Sync.state = "on"
    jet.eval()

    if i == 0:
        nu = jet.spectral_components.Sync.SED.nu
        tab["nu"] = nu

    tab[f"nuFnu (p={p:.1f})"] = jet.spectral_components.Sync.SED.nuFnu

    synch_pol_degree = jet.eval_synch_pol(nu.value / blob.delta_D * (1 + blob.z))[0]
    tab[f"pol_deg (p={p:.1f})"] = synch_pol_degree

# Plot
fig, axs = plt.subplots(2, 1, figsize=(8, 11))
for i, p in enumerate(spectral_indices):
    kwargs = {"label": f"p = {p:.1f}", "color": f"C{i}"}
    axs[0].loglog(nu, tab[f"nuFnu (p={p:.1f})"], **kwargs)
    axs[1].plot(nu, tab[f"pol_deg (p={p:.1f})"], **kwargs)
    axs[1].axhline(polarization_degree_power_law(p), ls="--", **kwargs)

ax = axs[0]
ax.set_xlabel(r"$\nu$ [Hz]", fontsize=12)
ax.set_ylabel(r"$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]", fontsize=12)
ax.legend(loc="upper left", fontsize=12)

ax = axs[1]
ax.set_xscale("log")
ax.grid(lw=0.5)
ax.set_xlabel(r"$\nu$ [Hz]", fontsize=12)
ax.set_ylabel(r"$\Pi_{\nu}$", fontsize=12)
ax.legend(loc="upper left", fontsize=12)

outname = "data/synch_polarization_jetset_1.3.0.ecsv"
tab.write(outname, format="ascii.ecsv", overwrite=True)
