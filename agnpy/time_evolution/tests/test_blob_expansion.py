import numpy as np
import astropy.units as u
import pytest
from astropy.constants import m_e, c
from copy import deepcopy
from agnpy import Blob, Synchrotron
from agnpy.time_evolution import BlobExpansion
from agnpy.time_evolution.time_evolution import TimeEvolution, synchrotron_loss, \
    ADIABATIC_EXPANSION_KEY, EXPANSION_DILUTION_KEY
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.utils.conversion import mec2


class TestBlobExpansion:

    @staticmethod
    def _make_blob(R_b=1e16 * u.cm, B=0.1 * u.G, p=2.1, k=1e-8 * u.Unit("cm-3"),
                   gamma_min=1e2, gamma_max=1e6):
        n_e = PowerLaw(k=k, p=p, gamma_min=gamma_min, gamma_max=gamma_max, mass=m_e)
        return Blob(R_b, B=B, n_e=n_e)

    def test_pure_adiabatic_expansion_matches_analytical_calculation(self):
        """ For R(t) = R_0 + v_exp * t, each electron energy follows gamma(t) = gamma_0 * R_0 / R(t) """
        blob = self._make_blob()
        r_0 = blob.R_b
        v_exp = 0.1 * c
        time = (r_0 / v_exp).to("s")  # R doubles over this time
        r_t = r_0 + (v_exp * time).to("cm")

        TimeEvolution(blob, time, expansion=BlobExpansion(v_exp),
                      max_energy_change_per_interval=0.001).evaluate()

        assert u.isclose(blob.R_b, r_t, rtol=1e-9)
        assert u.isclose(blob.n_e.gamma_min, 1e2 * (r_0 / r_t), rtol=0.001)
        assert u.isclose(blob.n_e.gamma_max, 1e6 * (r_0 / r_t), rtol=0.001)

    def test_total_number_of_electrons_is_conserved_during_expansion(self):
        """ The density dilution must exactly compensate the volume growth: N = n_e integral * V_b stays constant """
        blob = self._make_blob()
        electrons_before = blob.n_e.integrate() * blob.V_b
        v_exp = 0.1 * c
        time = (blob.R_b / v_exp).to("s")  # R doubles, V grows 8x

        TimeEvolution(blob, time, expansion=BlobExpansion(v_exp),
                      max_energy_change_per_interval=0.001).evaluate()
        electrons_after = blob.n_e.integrate() * blob.V_b

        assert u.isclose(electrons_before, electrons_after, rtol=0.01)

    def test_power_law_keeps_index_and_scales_amplitude_during_expansion(self):
        """ Under adiabatic losses and dilution, a power law k * gamma^-p evolves to the same index
            with amplitude k' = k * (R_0 / R_T)^(p+2) """
        p = 2.1
        k = 1e-8 * u.Unit("cm-3")
        blob = self._make_blob(p=p, k=k)
        r_0 = blob.R_b
        v_exp = 0.1 * c
        time = (r_0 / v_exp).to("s")
        r_t = r_0 + (v_exp * time).to("cm")

        TimeEvolution(blob, time, expansion=BlobExpansion(v_exp),
                      max_energy_change_per_interval=0.001).evaluate()

        expected_k = k * (r_0 / r_t) ** (p + 2)
        # sample the interior of the evolved distribution to avoid edge interpolation effects
        gammas = np.logspace(np.log10(blob.n_e.gamma_min * 2), np.log10(blob.n_e.gamma_max / 2), 20)
        assert u.allclose(blob.n_e(gammas), expected_k * gammas ** (-p), rtol=0.02)

    @pytest.mark.parametrize("magnetic_field_index", [0.0, 1.0, 2.0])
    def test_magnetic_field_scales_with_configured_index(self, magnetic_field_index):
        blob = self._make_blob()
        r_0 = blob.R_b
        b_0 = blob.B
        v_exp = 0.1 * c
        time = (0.5 * r_0 / v_exp).to("s")
        r_t = r_0 + (v_exp * time).to("cm")

        TimeEvolution(blob, time, expansion=BlobExpansion(v_exp, magnetic_field_index)).evaluate()

        # the multiplicative per-step update telescopes, so the result is exact up to float rounding
        assert u.isclose(blob.B, b_0 * (r_0 / r_t) ** magnetic_field_index, rtol=1e-9)

    def test_expansion_calculation_compared_with_calculation_split_into_two(self):
        """ Same as the split-run test in test_time_evolution.py, but with an expanding blob: the second run
            must continue from the radius and magnetic field reached by the first run """
        initial_n_e = BrokenPowerLaw(k=1e-8 * u.Unit("cm-3"), p1=1.9, p2=2.6, gamma_b=1e4,
                                     gamma_min=10, gamma_max=1e6, mass=m_e)
        r_0 = (100 * c * u.s).to(u.cm)
        expansion = BlobExpansion(0.3 * c, magnetic_field_index=1.0)

        blob1 = Blob(r_0, B=1 * u.G, n_e=deepcopy(initial_n_e))
        TimeEvolution(blob1, 60 * u.s, synchrotron_loss(Synchrotron(blob1)), expansion=expansion,
                      step_duration=3 * u.s).evaluate()

        blob2 = Blob(r_0, B=1 * u.G, n_e=deepcopy(initial_n_e))
        synch2 = synchrotron_loss(Synchrotron(blob2))
        TimeEvolution(blob2, 30 * u.s, synch2, expansion=expansion, step_duration=3 * u.s).evaluate()
        TimeEvolution(blob2, 30 * u.s, synch2, expansion=expansion, step_duration=3.5 * u.s).evaluate()

        assert u.isclose(blob1.R_b, blob2.R_b, rtol=1e-9)
        assert u.isclose(blob1.B, blob2.B, rtol=1e-9)
        gamma_min, gamma_max = blob1.n_e.gamma_min * 1.01, blob1.n_e.gamma_max * 0.99
        gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max))
        assert u.allclose(
            blob1.n_e.evaluate(gammas, 1, gamma_min, gamma_max),
            blob2.n_e.evaluate(gammas, 1, gamma_min, gamma_max),
            0.001)

    def test_heun_method_compared_to_euler_for_expansion(self):
        """ Heun method with 2x longer steps should still beat Euler on the analytically known
            adiabatic solution gamma(t) = gamma_0 * R_0 / R(t), and it should also conserve
            the total number of electrons much better, because the corrector recalculates
            the dilution at the predicted end-of-step state with width-aware count deposits """
        r_0 = (100 * c * u.s).to(u.cm)
        v_exp = 0.45 * c
        time = 200 * u.s
        r_t = r_0 + (v_exp * time).to("cm")

        def evolve(method, step_duration):
            blob = Blob(r_0, n_e=PowerLaw())
            initial_gamma = blob.gamma_e
            electrons_before = blob.n_e.integrate() * blob.V_b
            TimeEvolution(blob, time, expansion=BlobExpansion(v_exp), method=method,
                          step_duration=step_duration).evaluate()
            reversed_analytically = blob.gamma_e * (r_t / r_0).to_value("")
            gamma_error = np.average(np.abs((reversed_analytically - initial_gamma) / initial_gamma))
            electrons_after = blob.n_e.integrate() * blob.V_b
            electrons_error = np.abs((electrons_after - electrons_before) / electrons_before).to_value("")
            return gamma_error, electrons_error

        # Heun makes two rate evaluations per step, so compare it with 2x shorter Euler steps
        euler_gamma_error, euler_electrons_error = evolve("euler", 1 * u.s)
        heun_gamma_error, heun_electrons_error = evolve("heun", 2 * u.s)

        print("Average gamma error:")
        print("Euler (step length 1s)", f"{euler_gamma_error:.2E}")
        print("Heun (step length 2s)", f"{heun_gamma_error:.2E}")
        print("Electron count error")
        print("Euler (step length 1s)", f"{euler_electrons_error:.2E}")
        print("Heun (step length 2s)", f"{heun_electrons_error:.2E}")
        assert heun_gamma_error < euler_gamma_error
        assert heun_electrons_error < euler_electrons_error

    def test_expansion_with_subgroups_affects_all_groups(self):
        no_energy_change = lambda args: np.zeros(args.gamma.shape) * u.Unit("erg s-1")
        blob = self._make_blob()
        r_0 = blob.R_b
        electrons_before = blob.n_e.integrate() * blob.V_b
        v_exp = 0.1 * c
        time = (r_0 / v_exp).to("s")
        r_t = r_0 + (v_exp * time).to("cm")
        number_of_bins = 50
        gamma_array = np.logspace(2, 6, number_of_bins)
        half_half = np.full((2, number_of_bins), 0.5)

        result = TimeEvolution(blob, time, {"NoOp": no_energy_change},
                               expansion=BlobExpansion(v_exp),
                               subgroups=[["NoOp"], []],
                               subgroups_initial_density=half_half,
                               initial_gamma_array=gamma_array,
                               max_energy_change_per_interval=0.001).evaluate()

        # both groups experience the same adiabatic shift and dilution, so the split stays 50/50
        assert np.allclose(result.density_subgroups, 0.5, atol=0.01)
        assert u.isclose(blob.n_e.gamma_min, 1e2 * (r_0 / r_t), rtol=0.001)
        electrons_after = blob.n_e.integrate() * blob.V_b
        assert u.isclose(electrons_before, electrons_after, rtol=0.01)

    def test_expansion_rates_are_reported_in_results_and_callback(self):
        blob = self._make_blob()
        v_exp = 0.1 * c
        time = (0.2 * blob.R_b / v_exp).to("s")
        callback_results = []

        result = TimeEvolution(blob, time, expansion=BlobExpansion(v_exp),
                               distribution_change_callback=callback_results.append).evaluate()

        # the final rates were recalculated after the last radius advance, so they correspond to the final R
        expected_en_chg = -(result.gamma * mec2).to("erg") * v_exp / blob.R_b
        assert u.allclose(result.en_chg_rates[ADIABATIC_EXPANSION_KEY].to("erg s-1"), expected_en_chg, rtol=1e-6)
        expected_dilution = (-3 * v_exp / blob.R_b).to("s-1") * np.ones_like(result.gamma)
        assert u.allclose(result.rel_inj_rates[EXPANSION_DILUTION_KEY].to("s-1"), expected_dilution, rtol=1e-6)
        assert result.abs_inj_rates == {}
        # the callback must report the rates under the correct fields of TimeEvaluationResult
        assert EXPANSION_DILUTION_KEY in callback_results[-1].rel_inj_rates
        assert callback_results[-1].abs_inj_rates == {}

    def test_expansion_validation(self):
        blob = self._make_blob()
        with pytest.raises(ValueError):
            BlobExpansion(5 * u.cm)  # not a velocity
        with pytest.raises(ValueError):
            BlobExpansion(-1 * u.Unit("cm s-1"))  # contraction not supported
        with pytest.raises(ValueError):
            BlobExpansion(1.1 * c)  # faster than light
        with pytest.raises(ValueError):
            BlobExpansion(0.1 * c, magnetic_field_index=-1)
        with pytest.raises(ValueError):
            TimeEvolution(blob, 1 * u.s,
                          {ADIABATIC_EXPANSION_KEY: lambda args: None},
                          expansion=BlobExpansion(0.1 * c))
        with pytest.raises(ValueError):
            TimeEvolution(blob, 1 * u.s)  # no processes defined at all
        with pytest.raises(ValueError):
            TimeEvolution(blob, 1 * u.s, expansion=BlobExpansion(0.1 * c),
                          max_radius_change_for_cached_rates=0)
        with pytest.raises(ValueError):
            TimeEvolution(blob, 1 * u.s, expansion=BlobExpansion(0.1 * c),
                          max_radius_change_for_cached_rates=np.nan)
        # expansion combined with rate caching is supported
        TimeEvolution(blob, 1 * u.s, expansion=BlobExpansion(0.1 * c),
                      optimize_recalculating_slow_rates=True)

    def test_optimized_rates_match_unoptimized_with_expansion(self):
        """ Caching the rates of slowly-changing bins must not distort the results of an expanding-blob
            simulation: the internal expansion rates are always recalculated, and the staleness of the cached
            synchrotron rates is bounded by the step constraints and by max_radius_change_for_cached_rates """
        time = 2e6 * u.s
        v_exp = 0.05 * c

        def evolve(optimize):
            blob = self._make_blob(R_b=1e16 * u.cm, B=1 * u.G, gamma_max=1e7)
            synch = Synchrotron(blob)
            TimeEvolution(blob, time, synchrotron_loss(synch), expansion=BlobExpansion(v_exp),
                          optimize_recalculating_slow_rates=optimize).evaluate()
            return blob

        blob_optimized = evolve(True)
        blob_unoptimized = evolve(False)

        assert u.isclose(blob_optimized.R_b, blob_unoptimized.R_b, rtol=1e-9)
        assert u.isclose(blob_optimized.B, blob_unoptimized.B, rtol=1e-9)
        gammas = np.logspace(np.log10(blob_unoptimized.n_e.gamma_min * 1.05),
                             np.log10(blob_unoptimized.n_e.gamma_max * 0.95), 30)
        assert u.allclose(blob_optimized.n_e(gammas), blob_unoptimized.n_e(gammas), rtol=0.02)

    def test_optimization_reduces_rate_evaluations_with_expansion(self):
        """ The point of allowing expansion together with optimize_recalculating_slow_rates: when the highest
            energies force very small steps, the rates of the slowly-cooling bins should stay cached, because
            the radius barely changes during such steps. This test checks the speedup (the number of rate
            evaluations), so the expansion itself is chosen to be physically negligible """
        time = 100 * u.s
        v_exp = 0.001 * c

        def evolve(optimize):
            blob = self._make_blob(R_b=1e16 * u.cm, B=1 * u.G, gamma_max=1e7)
            base_loss = synchrotron_loss(Synchrotron(blob))
            evaluated_bins = [0]
            def counting_loss(args):
                evaluated_bins[0] += len(args.gamma)
                return base_loss(args)
            TimeEvolution(blob, time, counting_loss, expansion=BlobExpansion(v_exp),
                          optimize_recalculating_slow_rates=optimize).evaluate()
            return evaluated_bins[0]

        evaluations_optimized = evolve(True)
        evaluations_unoptimized = evolve(False)
        print("Synchrotron rate evaluations:", evaluations_optimized, "optimized,",
              evaluations_unoptimized, "unoptimized")
        assert evaluations_optimized < evaluations_unoptimized / 3
