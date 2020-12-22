# test on synchrotron module
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, h
from astropy.coordinates import Distance
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron, nu_synch_peak, epsilon_B
from agnpy.utils.math import trapz_loglog
from .utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
)


agnpy_dir = Path(__file__).parent.parent
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures
figures_dir = agnpy_dir.parent / "crosschecks/figures/synchrotron"
figures_dir.mkdir(parents=True, exist_ok=True)


# variables with _test are global and meant to be used in all tests
# here as a default we use the same parameters of Figure 7.4 in Dermer Menon 2009
spectrum_norm_test = 1e48 * u.Unit("erg")
p_test = 2.8
q_test = 0.2
gamma_0 = 1e3
gamma_min_test = 1e2
gamma_max_test = 1e5
pwl_dict_test = {
    "type": "PowerLaw",
    "parameters": {
        "p": p_test,
        "gamma_min": gamma_min_test,
        "gamma_max": gamma_max_test,
    },
}
# dictionary test for log parabola
lp_dict_test = {
    "type": "LogParabola",
    "parameters": {
        "p": p_test,
        "q": q_test,
        "gamma_0": gamma_0,
        "gamma_min": gamma_min_test,
        "gamma_max": gamma_max_test,
    },
}
# blob parameters
R_b_test = 1e16 * u.cm
B_test = 1 * u.G
z_test = Distance(1e27, unit=u.cm).z
delta_D_test = 10
Gamma_test = 10
pwl_blob_test = Blob(
    R_b_test,
    z_test,
    delta_D_test,
    Gamma_test,
    B_test,
    spectrum_norm_test,
    pwl_dict_test,
)
lp_blob_test = Blob(
    R_b_test,
    z_test,
    delta_D_test,
    Gamma_test,
    B_test,
    spectrum_norm_test,
    lp_dict_test,
)


class TestSynchrotron:
    """class grouping all tests related to the Synchrotron class"""

    @pytest.mark.parametrize("gamma_max, nu_range_max", [("1e5", 1e18), ("1e7", 1e22)])
    def test_synch_reference_sed(self, gamma_max, nu_range_max):
        """test agnpy synchrotron SED against the ones in Figure 7.4 of Dermer 
        Menon 2009"""
        # reference SED
        nu_ref, sed_ref = extract_columns_sample_file(
            f"{data_dir}/reference_seds/dermer_menon_2009/figure_7_4/synchrotron_gamma_max_{gamma_max}.txt",
            "Hz",
            "erg cm-2 s-1",
        )
        # agnpy
        # change the gamma_max in the blob
        pwl_dict_test["parameters"]["gamma_max"] = float(gamma_max)
        pwl_blob_test.set_spectrum(spectrum_norm_test, pwl_dict_test, "integral")
        # recompute the SED at the same ordinates of the reference figures
        synch = Synchrotron(pwl_blob_test)
        sed_agnpy = synch.sed_flux(nu_ref)
        # sed comparison plot
        nu_range = [1e10, nu_range_max] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_ref,
            sed_agnpy,
            "Figure 7.4, Dermer and Menon (2009)",
            "agnpy",
            "Synchrotron, " + r"$\gamma_{max} = $" + gamma_max,
            f"{figures_dir}/synch_comparison_figure_7_4_gamma_max_{gamma_max}_dermer_menon_2009.png",
            "sed",
            y_range=[1e-13, 1e-9],
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 25% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0, 0.25, nu_range)

    @pytest.mark.parametrize(
        "file_ref , spectrum_type, spectrum_parameters, figure_title, figure_path",
        [
            (
                f"{data_dir}/reference_seds/jetset/synch_ssa_pwl_jetset_1.1.2.txt",
                "PowerLaw",
                {"p": 2, "gamma_min": 2, "gamma_max": 1e6},
                "Self-Absorbed Synchrotron, power-law electron distribution",
                f"{figures_dir}/ssa_pwl_comparison_jetset_1.1.2.png",
            ),
            (
                f"{data_dir}/reference_seds/jetset/synch_ssa_bpwl_jetset_1.1.2.txt",
                "BrokenPowerLaw",
                {"p1": 2, "p2": 3, "gamma_b": 1e4, "gamma_min": 2, "gamma_max": 1e6},
                "Self-Absorbed Synchrotron, broken power-law electron distribution",
                f"{figures_dir}/ssa_bpwl_comparison_jetset_1.1.2.png",
            ),
            (
                f"{data_dir}/reference_seds/jetset/synch_ssa_lp_jetset_1.1.2.txt",
                "LogParabola",
                {"p": 2, "q": 0.4, "gamma_0": 1e4, "gamma_min": 2, "gamma_max": 1e6},
                "Self-Absorbed Synchrotron, log-parabola electron distribution",
                f"{figures_dir}/ssa_lp_comparison_jetset_1.1.2.png",
            ),
        ],
    )
    def test_ssa_reference_sed(
        self, file_ref, spectrum_type, spectrum_parameters, figure_title, figure_path,
    ):
        """test SSA SED generated by a given electron distribution against the 
        ones generated with jetset version 1.1.2, via jetset_ssa_sed.py script"""
        # reference SED
        nu_ref, sed_ref = extract_columns_sample_file(file_ref, "Hz", "erg cm-2 s-1")
        # same parameters used to produce the jetset SED
        spectrum_norm = 1e2 * u.Unit("cm-3")
        spectrum_dict = {"type": spectrum_type, "parameters": spectrum_parameters}
        blob = Blob(
            R_b=5e15 * u.cm,
            z=0.1,
            delta_D=10,
            Gamma=10,
            B=0.1 * u.G,
            spectrum_norm=spectrum_norm,
            spectrum_dict=spectrum_dict,
        )
        # recompute the SED at the same ordinates where the figure was sampled
        ssa = Synchrotron(blob, ssa=True)
        sed_agnpy = ssa.sed_flux(nu_ref)
        # sed comparison plot, we will check between 10^(11) and 10^(19) Hz
        nu_range = [1e11, 1e19] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_ref,
            sed_agnpy,
            "jetset 1.1.2",
            "agnpy",
            figure_title,
            figure_path,
            "sed",
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 5% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0, 0.05, nu_range)

    def test_synch_delta_sed(self):
        """check that in a given frequency range the full synchrotron SED coincides
        with the delta function approximation"""
        nu = np.logspace(10, 20) * u.Hz
        synch = Synchrotron(lp_blob_test)
        sed_full = synch.sed_flux(nu)
        sed_delta = synch.sed_flux_delta_approx(nu)
        # range of comparison
        nu_range = [1e12, 1e17] * u.Hz
        make_comparison_plot(
            nu,
            sed_full,
            sed_delta,
            "full integration",
            "delta function approximation",
            "Synchrotron",
            f"{figures_dir}/synch_comparison_delta_aprproximation.png",
            "sed",
            [1e-16, 1e-8],
            nu_range.to_value("Hz"),
        )
        # requires that the delta approximation SED points deviate less than 10%
        assert check_deviation(nu, sed_delta, sed_full, 0, 0.1, nu_range)

    def test_sed_integration_methods(self):
        """test different integration methods agains each other
        simpole trapezoidal rule vs trapezoidal rule in log-log space
        """
        nu = np.logspace(8, 22) * u.Hz
        synch_trapz = Synchrotron(pwl_blob_test, integrator=np.trapz)
        synch_trapz_loglog = Synchrotron(pwl_blob_test, integrator=trapz_loglog)
        sed_synch_trapz = synch_trapz.sed_flux(nu)
        sed_synch_trapz_loglog = synch_trapz_loglog.sed_flux(nu)
        make_comparison_plot(
            nu,
            sed_synch_trapz,
            sed_synch_trapz_loglog,
            "trapezoidal integration",
            "trapezoidal log-log integration",
            "Synchrotron",
            f"{figures_dir}/synch_comparison_integration_methods.png",
            "sed",
        )
        # requires that the SED points deviate less than 1%
        assert check_deviation(nu, sed_synch_trapz_loglog, sed_synch_trapz, 0, 0.01)

    def test_nu_synch_peak(self):
        gamma = 100
        nu_synch = nu_synch_peak(B_test, gamma).to_value("Hz")
        assert np.isclose(nu_synch, 27992489872.33304, atol=0)

    def test_epsilon_B(self):
        assert np.isclose(epsilon_B(B_test), 2.2655188038060715e-14, atol=0)
