# wrap agnpy SED computation via Gammapy's SpectralModel
import numpy as np
import astropy.units as u
from astropy.constants import c, k_B
from astropy.coordinates import Distance
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SpectralModel
from ..utils.conversion import mec2
from ..synchrotron import Synchrotron
from ..compton import SynchrotronSelfCompton, ExternalCompton
from ..targets import SSDisk, RingDustTorus
from .core import (
    get_spectral_parameters_from_n_e,
    get_emission_region_parameters,
    get_targets_parameters,
)


gamma_size = 400
gamma_to_integrate = np.logspace(1, 9, gamma_size)


def add_systematic_errors_flux_points(flux_points, syst):
    """Add the systematic error on the flux points in a given energy range.
    We symply sum the systematic error in quadrature with the statystical one.
    The systematic error is given as a percentage of the flux value.

    Parameters
    ----------
    flux_points : `~gammapy.estimators.FluxPoints`
        Gammapy's flux points
    syst : float
        systematic error as a percentage of the dnde flux
    """
    dnde_err_syst = syst * flux_points.dnde.data
    # sum in quadrature with the stat error
    dnde_errn_tot = np.sqrt(flux_points.dnde_errn.data ** 2 + dnde_err_syst ** 2)
    dnde_errp_tot = np.sqrt(flux_points.dnde_errp.data ** 2 + dnde_err_syst ** 2)
    # the attributes we have to change is the norm_errn and norm_errp
    flux_points.norm_errn.data = dnde_errn_tot / flux_points.dnde_ref.data
    flux_points.norm_errp.data = dnde_errp_tot / flux_points.dnde_ref.data


def _sort_spectral_parameters(spectral_pars_names, **kwargs):
    """All the model parameters will be passed as **kwargs by
    SpectralModel.evaluate(). This function helps sort out those related to the
    particle energy distribution.
    Parameters are returned as a simple list.
    """
    args = [
        10 ** (kwargs[key].value) if key.startswith("log10_") else kwargs[key].value
        for key in spectral_pars_names
    ]
    # add unit to the norm, which is always the first one
    args[0] *= u.Unit("cm-3")
    return args


def _sort_emission_region_parameters(scenario, **kwargs):
    """All the model parameters will be passed as **kwargs by
    SpectralModel.evaluate(). This function helps sort out those related to the
    emission region.
    """
    z = kwargs["z"]
    delta_D = kwargs["delta_D"]
    B = 10 ** kwargs["log10_B"] * u.G
    t_var = kwargs["t_var"]

    # compute the luminosity distance and the size of the emission region
    d_L = Distance(z=z).to("cm")
    R_b = (c * t_var * delta_D / (1 + z)).to("cm")

    if scenario == "ssc":
        return z, d_L, delta_D, B, R_b

    if scenario == "ec":
        # there are two additional emission region parameters in case of EC
        mu_s = kwargs["mu_s"]
        r = 10 ** kwargs["log10_r"] * u.cm

        return z, d_L, delta_D, B, R_b, mu_s, r


def _sort_disk_parameters(**kwargs):
    """Same as the functions above, but for the disk."""
    L_disk = kwargs.pop("L_disk")
    M_BH = kwargs.pop("M_BH")
    m_dot = kwargs.pop("m_dot")
    R_in = kwargs.pop("R_in")
    R_out = kwargs.pop("R_out")

    return L_disk, M_BH, m_dot, R_in, R_out


def _sort_blr_parameters(**kwargs):
    """Same as the functions above, but for the BLR."""
    xi_line = kwargs["xi_line"]
    lambda_line = kwargs["lambda_line"]
    epsilon_line = (lambda_line.to("erg", equivalencies=u.spectral()) / mec2).to_value(
        ""
    )
    R_line = kwargs["R_line"]

    return xi_line, epsilon_line, R_line


def _sort_blr_parameters(**kwargs):
    """Same as the functions above, but for the BLR."""
    xi_line = kwargs["xi_line"]
    lambda_line = kwargs["lambda_line"]
    R_line = kwargs["R_line"]
    epsilon_line = lambda_line.to("erg", equivalencies=u.spectral()) / mec2

    return xi_line, epsilon_line.to_value(""), R_line


def _sort_dt_parameters(**kwargs):
    xi_dt = kwargs["xi_dt"]
    T_dt = kwargs["T_dt"]
    R_dt = kwargs["R_dt"]
    epsilon_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

    return xi_dt, epsilon_dt, T_dt, R_dt


class SynchrotronSelfComptonSpectralModel(SpectralModel):

    tag = ["SynchrotronSelfComptonSpectralModel"]

    def __init__(self, n_e, ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron and SSC radiation.

        Parameters
        ----------
        n_e : `~agnpy.spectra.ElectronDistribution`
            electron distribution to be used for this modelling
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~gammapy.modeling.models.SpectralModel`
        """
        self._n_e = n_e
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_n_e(self._n_e, backend="gammapy")
        self._spectral_pars_names = list(spectral_pars.keys())

        # parameters of the emission region
        emission_region_pars = get_emission_region_parameters("ssc", backend="gammapy")
        self._emission_region_pars_names = list(emission_region_pars.keys())

        # group the model parameters, add the norm at the bottom of the list
        norm = Parameter("norm", 1, min=0.1, max=10, is_norm=True, frozen=True)
        self.default_parameters = Parameters(
            [*spectral_pars.values(), *emission_region_pars.values(), norm]
        )

        super().__init__()

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self._spectral_pars_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self._emission_region_pars_names)

    def evaluate(self, energy, **kwargs):
        """Evaluate the SED model.
        NOTE: All the model parameters will be passed as kwargs by
        SpectralModel.evaluate()."""

        nu = energy.to("Hz", equivalencies=u.spectral())

        args = _sort_spectral_parameters(self._spectral_pars_names, **kwargs)
        z, d_L, delta_D, B, R_b = _sort_emission_region_parameters("ssc", **kwargs)

        # evaluate the synch. and SSC SEDs
        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
        )
        sed = sed_synch + sed_ssc

        # to avoid the problem pointed out in
        # https://github.com/cosimoNigro/agnpy/issues/117
        # we can do here something like
        # sed = sed.reshape(energy.shape)
        # this is done in Gammapy's `NaimaSpectralModel`
        # https://github.com/gammapy/gammapy/blob/master/gammapy/modeling/models/spectral.py#L2119

        # gammapy requires a differential flux in input
        return (sed / energy ** 2).to("1 / (cm2 eV s)")


class ExternalComptonSpectralModel(SpectralModel):

    tag = ["ExternalComptonSpectralModel"]

    def __init__(self, n_e, targets, ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron, SSC, and EC on a
        list of targets.

        Parameters
        ----------
        n_e : `~agnpy.spectra.ElectronDistribution`
            electron distribution to be used for this modelling
        targets : list of strings ["blr", "dt"]
            targets to be considered for external Compton
            typically we consider the BLR or the DT or both, the EC on disk
            is not considered as it is subdominant at distances >> R_out
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~gammapy.modeling.models.SpectralModel`
        """

        self._n_e = n_e
        self.targets = targets
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_n_e(self._n_e, backend="gammapy")
        self._spectral_pars_names = list(spectral_pars.keys())

        # parameters of the emission region
        emission_region_pars = get_emission_region_parameters("ec", backend="gammapy")
        self._emission_region_pars_names = list(emission_region_pars.keys())

        # parameters of the targets
        targets_pars = get_targets_parameters(self.targets, backend="gammapy")
        self._targets_pars_names = list(targets_pars.keys())

        # group the model parameters, add the norm at the bottom of the list
        norm = Parameter("norm", 1, min=0.1, max=10, is_norm=True, frozen=True)
        self.default_parameters = Parameters(
            [
                *spectral_pars.values(),
                *emission_region_pars.values(),
                *targets_pars.values(),
                norm,
            ]
        )

        super().__init__()

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self._spectral_pars_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self._emission_region_pars_names)

    @property
    def targets_parameters(self):
        """Select all the parameters related to the targets for EC."""
        return self.parameters.select(self._targets_pars_names)

    def evaluate(self, energy, **kwargs):
        """Evaluate the SED model.
        NOTE: All the model parameters will be passed as kwargs by
        SpectralModel.evaluate()."""

        nu = energy.to("Hz", equivalencies=u.spectral())

        args = _sort_spectral_parameters(self._spectral_pars_names, **kwargs)
        z, d_L, delta_D, B, R_b, mu_s, r = _sort_emission_region_parameters(
            "ec", **kwargs
        )

        # evaluate the synch. and SSC SEDs
        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
        )
        sed = sed_synch + sed_ssc

        # add the disk thermal components
        L_disk, M_BH, m_dot, R_in, R_out = _sort_disk_parameters(**kwargs)
        sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
            nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        sed += sed_bb_disk

        # add the EC components
        if "blr" in self.targets:
            xi_line, epsilon_line, R_line = _sort_blr_parameters(**kwargs)
            sed_ec_blr = ExternalCompton.evaluate_sed_flux_blr(
                nu,
                z,
                d_L,
                delta_D,
                mu_s,
                R_b,
                L_disk,
                xi_line,
                epsilon_line,
                R_line,
                r,
                self._n_e,
                *args,
                gamma=gamma_to_integrate
            )
            sed += sed_ec_blr

        if "dt" in self.targets:
            xi_dt, epsilon_dt, T_dt, R_dt = _sort_dt_parameters(**kwargs)
            sed_ec_dt = ExternalCompton.evaluate_sed_flux_dt(
                nu,
                z,
                d_L,
                delta_D,
                mu_s,
                R_b,
                L_disk,
                xi_dt,
                epsilon_dt,
                R_dt,
                r,
                self._n_e,
                *args,
                gamma=gamma_to_integrate
            )
            sed += sed_ec_dt
            # add the thermal emission of the DT as well
            sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
                nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
            )
            sed += sed_bb_dt

        # eventual reshaping
        # we can do here something like
        # sed = sed.reshape(energy.shape)
        # see the same comment in the SSC model

        return (sed / energy ** 2).to("1 / (cm2 eV s)")
