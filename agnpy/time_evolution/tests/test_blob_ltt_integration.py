from copy import deepcopy

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import c, m_e

from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.spectra import PowerLaw
from agnpy.time_evolution import ltt_integrator_constant_radius

C_CGS = c.to_value("cm/s")
SED_UNIT = u.Unit("erg / (cm2 s)")


def make_blob(R_b=1e16 * u.cm, B=0.1 * u.G):
    n_e = PowerLaw(k=1e-8 * u.Unit("cm-3"), p=2.1, gamma_min=1e2, gamma_max=1e6, mass=m_e)
    return Blob(R_b, B=B, n_e=n_e)


def flat_sed(value):
    """A sed_flux_fn with a constant, frequency-independent flux."""
    return lambda blob, nu: np.full(len(nu), value) * SED_UNIT


def kernel_integral(window):
    tau, W = window.kernel
    return np.trapz(W.to_value("1/s"), tau.to_value("s"))


class TestKernel:
    """The geometric kernel: shape, limits and normalisation."""

    def test_shape_and_normalisation(self):
        R = 1e16 * u.cm
        integrator = ltt_integrator_constant_radius(R, [1e15] * u.Hz, kernel_points_size=2001)
        window = integrator.for_time(0 * u.s)
        tau, W = window.kernel

        tau_max = (R / c).to("s")
        assert u.isclose(tau[0], -tau_max, rtol=1e-12)
        assert u.isclose(tau[-1], tau_max, rtol=1e-12)
        assert len(tau) == 2001 == integrator.kernel_points_size

        # normalised to 1, peaked at 3c/4R, vanishing at both ends
        assert np.isclose(kernel_integral(window), 1.0, rtol=1e-5)
        assert np.isclose(W[1000].to_value("1/s"), 0.75 * C_CGS / R.to_value("cm"), rtol=1e-12)
        assert W[0].to_value("1/s") == 0
        assert W[-1].to_value("1/s") == 0

    def test_kernel_is_symmetric(self):
        integrator = ltt_integrator_constant_radius(1e16 * u.cm, [1e15] * u.Hz)
        _, W = integrator.for_time(3e5 * u.s).kernel
        assert np.allclose(W.to_value("1/s"), W.to_value("1/s")[::-1])

    def test_kernel_is_independent_of_time(self):
        """With a fixed radius the kernel is the same at every time, and is computed once."""
        integrator = ltt_integrator_constant_radius(1e16 * u.cm, [1e15] * u.Hz)
        tau_a, W_a = integrator.for_time(0 * u.s).kernel
        tau_b, W_b = integrator.for_time(1e7 * u.s).kernel
        assert np.allclose(tau_a.to_value("s"), tau_b.to_value("s"))
        assert np.allclose(W_a.to_value("1/s"), W_b.to_value("1/s"))

    def test_exposed_kernel_cannot_corrupt_the_cached_one(self):
        integrator = ltt_integrator_constant_radius(1e16 * u.cm, [1e15] * u.Hz)
        tau, W = integrator.for_time(0 * u.s).kernel
        tau[:] = 0
        W[:] = 0
        tau_again, W_again = integrator.for_time(0 * u.s).kernel
        assert np.any(tau_again.to_value("s") != 0)
        assert np.isclose(kernel_integral(integrator.for_time(0 * u.s)), 1.0, rtol=1e-3)


class TestWindow:
    """The blob-state span reported by for_time."""

    def test_window_is_symmetric_about_the_central_time(self):
        R, t0 = 1e16 * u.cm, 5e5 * u.s
        integrator = ltt_integrator_constant_radius(R, [1e15] * u.Hz)
        window = integrator.for_time(t0)

        assert u.isclose(window.central_time, t0)
        assert u.isclose(window.end_time - window.central_time, (R / c).to("s"), rtol=1e-12)
        assert u.isclose(window.central_time - window.start_time, (R / c).to("s"), rtol=1e-12)

    def test_for_time_zero_reports_the_required_lead(self):
        """for_time is a pure geometric query: it must work at t=0 and report a negative start."""
        R = 1e16 * u.cm
        integrator = ltt_integrator_constant_radius(R, [1e15] * u.Hz)

        lead = -integrator.for_time(0 * u.s).start_time
        assert lead > 0
        assert u.isclose(lead, (R / c).to("s"), rtol=1e-12)


class TestCalcSed:
    """Integration of blob states over the window."""

    def test_constant_state_reproduces_the_plain_sed(self):
        """With a frozen blob the smearing must be a no-op: the kernel integrates to 1."""
        nu = np.logspace(11, 24, 6) * u.Hz
        R = 1e16 * u.cm
        blob = make_blob(R)
        integrator = ltt_integrator_constant_radius(R, nu)

        t0 = 2 * (R / c).to("s")
        window = integrator.for_time(t0)
        times = np.linspace(0, 4 * (R / c).to_value("s"), 4) * u.s
        blobs = [deepcopy(blob) for _ in times]

        sed = window.calc_sed(blobs, times)
        expected = Synchrotron(blob).sed_flux(nu) + SynchrotronSelfCompton(blob).sed_flux(nu)

        assert sed.unit.is_equivalent(SED_UNIT)
        assert u.allclose(sed, expected, rtol=1e-3)

    def test_state_linear_in_time_integrates_to_the_central_value(self):
        """The kernel is symmetric, so its first moment vanishes."""
        nu = [1e15] * u.Hz
        R = 1e16 * u.cm
        light_crossing = (R / c).to_value("s")
        offset, slope = 3.0, 2e-3

        def sed_flux_fn(blob, freq):
            return np.full(len(freq), offset + slope * blob.marker_time) * SED_UNIT

        integrator = ltt_integrator_constant_radius(R, nu, sed_flux_fn=sed_flux_fn)
        t0 = 2 * light_crossing * u.s
        window = integrator.for_time(t0)

        times = np.linspace(0, 4 * light_crossing, 40) * u.s
        blobs = []
        for t in times:
            blob = make_blob(R)
            blob.marker_time = t.to_value("s")
            blobs.append(blob)

        sed = window.calc_sed(blobs, times)
        assert np.isclose(
            sed[0].to_value(SED_UNIT), offset + slope * t0.to_value("s"), rtol=1e-3
        )

    def test_smearing_lags_a_step_change(self):
        """A step in the state is seen smeared over the whole light-crossing window."""
        nu = [1e15] * u.Hz
        R = 1e16 * u.cm
        lc = (R / c).to_value("s")

        def sed_flux_fn(blob, freq):
            return np.full(len(freq), 0.0 if blob.marker_time < 5 * lc else 1.0) * SED_UNIT

        integrator = ltt_integrator_constant_radius(R, nu, sed_flux_fn=sed_flux_fn)
        times = np.linspace(0, 10 * lc, 400) * u.s
        blobs = []
        for t in times:
            blob = make_blob(R)
            blob.marker_time = t.to_value("s")
            blobs.append(blob)

        def flux(t0_over_lc):
            return integrator.for_time(t0_over_lc * lc * u.s).calc_sed(
                blobs, times
            )[0].to_value(SED_UNIT)

        # fully before the step, halfway through it, fully after.
        # The tolerance on the plateau is set by the kernel quadrature error (~4e-6 at the
        # default 500 points), not by the smearing itself.
        assert np.isclose(flux(3.0), 0.0, atol=1e-6)
        assert np.isclose(flux(5.0), 0.5, rtol=2e-2)
        assert np.isclose(flux(7.0), 1.0, rtol=1e-4)
        # monotonically rising across the transition
        rising = [flux(x) for x in np.linspace(4.0, 6.0, 9)]
        assert all(a <= b + 1e-12 for a, b in zip(rising, rising[1:]))


class TestRadiusValidation:
    """calc_sed refuses states whose R_b disagrees with the integrator's radius."""

    def _setup(self, snapshot_radius):
        R = 1e16 * u.cm
        integrator = ltt_integrator_constant_radius(R, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0))
        lc = (R / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 4 * lc, 5) * u.s
        blobs = [make_blob(snapshot_radius) for _ in times]
        return window, blobs, times

    def test_consistent_radii_pass(self):
        window, blobs, times = self._setup(1e16 * u.cm)
        sed = window.calc_sed(blobs, times)
        assert np.all(np.isfinite(sed.to_value(SED_UNIT)))

    def test_mismatched_radius_raises(self):
        window, blobs, times = self._setup(2e16 * u.cm)
        with pytest.raises(ValueError, match="integrator was built for"):
            window.calc_sed(blobs, times)

    def test_calc_sed_does_not_modify_the_snapshots(self):
        window, blobs, times = self._setup(1e16 * u.cm)
        before = [blob.R_b.copy() for blob in blobs]
        window.calc_sed(blobs, times)
        for blob, original in zip(blobs, before):
            assert u.isclose(blob.R_b, original, rtol=0, atol=0 * u.cm)


class TestCoverage:
    """Windows that reach outside the supplied snapshots."""

    def _integrator(self):
        return ltt_integrator_constant_radius(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0)
        )

    def test_missing_states_before_the_window_raise(self):
        integrator = self._integrator()
        window = integrator.for_time(0 * u.s)  # start_time < 0
        times = np.linspace(0, 1e6, 10) * u.s
        blobs = [make_blob() for _ in times]

        with pytest.raises(ValueError, match="missing before the window"):
            window.calc_sed(blobs, times)

    def test_missing_states_after_the_window_raise(self):
        integrator = self._integrator()
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 2.5 * lc, 10) * u.s  # ends before end_time
        blobs = [make_blob() for _ in times]

        with pytest.raises(ValueError, match="missing after the window"):
            window.calc_sed(blobs, times)

    def test_no_state_before_first_zero_fills_instead(self):
        """Opting into 'the blob had not formed yet' contributes zero, not a held edge value."""
        integrator = self._integrator()
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(0 * u.s)
        times = np.linspace(0, 2 * lc, 60) * u.s
        blobs = [make_blob() for _ in times]

        sed = window.calc_sed(blobs, times, no_state_before_first=True)

        # only the tau > 0 half contributes, and the kernel is symmetric, so ~half the flux
        assert np.isclose(sed[0].to_value(SED_UNIT), 0.5, rtol=1e-2)

    def test_the_documented_preflight_makes_the_window_covered(self):
        integrator = self._integrator()
        lead = -integrator.for_time(0 * u.s).start_time
        # start the "simulation" `lead` early, recording times on the shifted clock
        times = np.linspace(-lead.to_value("s"), 1e6, 60) * u.s
        blobs = [make_blob() for _ in times]

        sed = integrator.for_time(0 * u.s).calc_sed(blobs, times)
        assert np.isclose(sed[0].to_value(SED_UNIT), 1.0, rtol=1e-3)


class TestSedCache:
    @staticmethod
    def _counting_integrator(calls):
        def counting_sed(blob, nu):
            calls.append(id(blob))
            return np.full(len(nu), 1.0) * SED_UNIT

        return ltt_integrator_constant_radius(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=counting_sed
        )

    def test_each_snapshot_is_evaluated_once(self):
        calls = []
        integrator = self._counting_integrator(calls)
        lc = (1e16 * u.cm / c).to_value("s")

        times = np.linspace(0, 10 * lc, 9) * u.s
        blobs = [make_blob() for _ in times]

        # two overlapping windows over the same growing snapshot list
        integrator.for_time(2 * lc * u.s).calc_sed(blobs[:6], times[:6])
        assert len(calls) == 6
        integrator.for_time(5 * lc * u.s).calc_sed(blobs, times)
        # only the three new snapshots are evaluated
        assert len(calls) == 9

    def test_replaced_snapshot_is_recomputed(self):
        calls = []
        integrator = self._counting_integrator(calls)
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)

        times = np.linspace(0, 5 * lc, 6) * u.s
        blobs = [make_blob() for _ in times]
        window.calc_sed(blobs, times)
        assert len(calls) == 6

        # same times, but one state replaced by a different object -> must be re-evaluated
        blobs[3] = make_blob()
        window.calc_sed(blobs, times)
        assert len(calls) == 7


class TestValidation:
    def _window(self):
        integrator = ltt_integrator_constant_radius(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0)
        )
        return integrator.for_time(0 * u.s)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            self._window().calc_sed([make_blob(), make_blob()], [0, 1, 2] * u.s)

    def test_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            self._window().calc_sed([], [] * u.s)

    def test_non_monotonic_times(self):
        blobs = [make_blob() for _ in range(3)]
        with pytest.raises(ValueError, match="strictly increasing"):
            self._window().calc_sed(blobs, [0, 2, 1] * u.s)

    def test_non_positive_radius(self):
        with pytest.raises(ValueError, match="positive finite length"):
            ltt_integrator_constant_radius(0 * u.cm, [1e15] * u.Hz)

    def test_too_few_kernel_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            ltt_integrator_constant_radius(1e16 * u.cm, [1e15] * u.Hz, kernel_points_size=1)


class TestSingleSnapshot:
    """
    A single snapshot cannot cover a window on its own, so the frozen-state branch is only
    reachable by explicitly declaring that nothing emitted before it.
    """

    def test_single_snapshot_cannot_cover_a_window(self):
        integrator = ltt_integrator_constant_radius(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0)
        )
        with pytest.raises(ValueError, match="missing before the window"):
            integrator.for_time(0 * u.s).calc_sed([make_blob()], [0] * u.s)

    def test_frozen_state_branch(self):
        R = 1e16 * u.cm
        lc = (R / c).to_value("s")
        integrator = ltt_integrator_constant_radius(R, [1e15] * u.Hz, sed_flux_fn=flat_sed(7.0))
        window = integrator.for_time(0 * u.s)
        # the only snapshot sits at end_time, so just the last grid point contributes
        times = [lc] * u.s

        sed = window.calc_sed([make_blob(R)], times, no_state_before_first=True)

        tau, W = window.kernel
        tau_s, W_cgs = tau.to_value("s"), W.to_value("1/s")
        weights = np.where(tau_s < times[0].to_value("s"), 0.0, W_cgs)
        assert np.isclose(sed[0].to_value(SED_UNIT), 7.0 * np.trapz(weights, tau_s), rtol=1e-12)
