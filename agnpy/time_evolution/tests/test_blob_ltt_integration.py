from copy import deepcopy

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import c, m_e

from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.spectra import PowerLaw
from agnpy.time_evolution import BlobLTTIntegrator, TimeEvolution, synchrotron_loss
from agnpy.time_evolution.blob_ltt_integration import _constant_kernel_cgs

C_CGS = c.to_value("cm/s")
SED_UNIT = u.Unit("erg / (cm2 s)")


def make_blob(R_b=1e16 * u.cm, B=0.1 * u.G):
    n_e = PowerLaw(k=1e-8 * u.Unit("cm-3"), p=2.1, gamma_min=1e2, gamma_max=1e6, mass=m_e)
    return Blob(R_b, B=B, n_e=n_e)


def flat_sed(value):
    """A sed_flux_fn with a constant, frequency-independent flux."""
    return lambda blob, nu: np.full(len(nu), value) * SED_UNIT


class TestKernel:
    """The geometric kernel: shape, limits and normalisation."""

    def test_shape_and_normalisation(self):
        R_cm = 1e16
        n = 41
        tau, W = _constant_kernel_cgs(R_cm, n)

        tau_max = R_cm / C_CGS
        assert np.isclose(tau[0], -tau_max, rtol=1e-12)
        assert np.isclose(tau[-1], tau_max, rtol=1e-12)
        assert len(tau) == len(W) == n

        # normalised to 1, peaked at 3c/4R, vanishing at both ends
        assert np.isclose(np.trapz(W, tau), 1.0, rtol=1e-3)
        assert np.isclose(W[n // 2], 0.75 * C_CGS / R_cm, rtol=1e-12)
        assert W[0] == 0
        assert W[-1] == 0

    def test_kernel_is_symmetric(self):
        tau, W = _constant_kernel_cgs(1e16, 51)
        assert np.allclose(W, W[::-1])
        assert np.allclose(tau, -tau[::-1])

    @pytest.mark.parametrize("n", [10, 50, 200, 2000])
    def test_quadrature_error_is_second_order(self, n):
        """
        The kernel integrates to 1 analytically. Composite trapezoid on a parabola has an exact
        error of (b-a) h^2 |f''| / 12, which here works out to 1/(n-1)**2 -- so the shortfall is
        fully predictable, and this pins both the normalisation and the convergence rate.
        """
        tau, W = _constant_kernel_cgs(1e16, n)
        assert np.isclose(np.trapz(W, tau), 1.0 - 1.0 / (n - 1) ** 2, rtol=1e-9)


class TestWindow:
    """The blob-state span reported by for_time."""

    def test_window_is_centred_on_the_requested_time_and_reaches_negative_start(self):
        R = 1e16 * u.cm
        t = 5e2 * u.s  # early enough that the window reaches back before 0
        integrator = BlobLTTIntegrator(R, [1e15] * u.Hz)
        window = integrator.for_time(t)

        assert window.start_time < 0 * u.s
        assert u.isclose(window.end_time - t, (R / c).to("s"), rtol=1e-12)
        assert u.isclose(t - window.start_time, (R / c).to("s"), rtol=1e-12)

    def test_window_width_is_independent_of_time(self):
        """With a fixed radius every window spans the same 2R/c, wherever it is centred."""
        integrator = BlobLTTIntegrator(1e16 * u.cm, [1e15] * u.Hz)
        widths = [
            (integrator.for_time(t).end_time - integrator.for_time(t).start_time)
            for t in [0, 3e5, 1e7] * u.s
        ]
        assert u.allclose(u.Quantity(widths), 2 * (1e16 * u.cm / c).to("s"), rtol=1e-12)


class TestCalcSed:
    """Integration of blob states over the window."""

    def test_constant_state_reproduces_the_plain_sed(self):
        """With a frozen blob the smearing must be a no-op"""
        nu = np.logspace(11, 24, 6) * u.Hz
        R = 1e16 * u.cm
        blob = make_blob(R)
        integrator = BlobLTTIntegrator(R, nu)

        t0 = 2 * (R / c).to("s")
        window = integrator.for_time(t0)
        times = np.linspace(0, 4 * (R / c).to_value("s"), 4) * u.s
        snapshots = [(t, deepcopy(blob)) for t in times]

        sed = window.calc_sed(snapshots)
        expected = Synchrotron(blob).sed_flux(nu) + SynchrotronSelfCompton(blob).sed_flux(nu)

        assert u.allclose(sed, expected, rtol=1e-3)

    def test_state_linear_in_time_integrates_to_the_central_value(self):
        """The kernel is symmetric, so its first moment vanishes."""
        nu = [1e15] * u.Hz
        R = 1e16 * u.cm
        light_crossing = (R / c).to_value("s")
        offset, slope = 3.0, 2e-3

        def sed_flux_fn(blob, freq):
            return np.full(len(freq), offset + slope * blob.marker_time) * SED_UNIT

        integrator = BlobLTTIntegrator(R, nu, sed_flux_fn=sed_flux_fn)
        t0 = 2 * light_crossing * u.s
        window = integrator.for_time(t0)

        times = np.linspace(0, 4 * light_crossing, 40) * u.s
        snapshots = []
        for t in times:
            blob = make_blob(R)
            blob.marker_time = t.to_value("s")
            snapshots.append((t, blob))

        sed = window.calc_sed(snapshots)
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

        integrator = BlobLTTIntegrator(R, nu, sed_flux_fn=sed_flux_fn)
        times = np.linspace(0, 10 * lc, 400) * u.s
        snapshots = []
        for t in times:
            blob = make_blob(R)
            blob.marker_time = t.to_value("s")
            snapshots.append((t, blob))

        def flux(t0_over_lc):
            return integrator.for_time(t0_over_lc * lc * u.s).calc_sed(
                snapshots
            )[0].to_value(SED_UNIT)

        # fully before the step, halfway through it, fully after.
        # The tolerance on the plateau is set by the kernel quadrature error (~4e-4 at the
        # default 50 points), not by the smearing itself.
        assert np.isclose(flux(3.0), 0.0, atol=1e-6)
        assert np.isclose(flux(5.0), 0.5, rtol=2e-2)
        assert np.isclose(flux(7.0), 1.0, rtol=1e-3)
        # monotonically rising across the transition
        rising = [flux(x) for x in np.linspace(4.0, 6.0, 9)]
        assert np.all(np.diff(rising) >= -1e-12)


class TestRadiusValidation:
    """calc_sed refuses states whose R_b disagrees with the integrator's radius."""

    def _setup(self, snapshot_radius):
        R = 1e16 * u.cm
        integrator = BlobLTTIntegrator(R, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0))
        lc = (R / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 4 * lc, 5) * u.s
        snapshots = [(t, make_blob(snapshot_radius)) for t in times]
        return window, snapshots

    def test_consistent_radii_pass(self):
        window, snapshots = self._setup(1e16 * u.cm)
        sed = window.calc_sed(snapshots)
        assert np.all(np.isfinite(sed.to_value(SED_UNIT)))

    def test_mismatched_radius_raises(self):
        window, snapshots = self._setup(2e16 * u.cm)
        with pytest.raises(ValueError, match="integrator was built for"):
            window.calc_sed(snapshots)

class TestCoverage:
    """The snapshots must bracket the window: interpolation only, never extrapolation."""

    def _integrator(self):
        return BlobLTTIntegrator(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0)
        )

    def test_missing_states_before_the_window_raise(self):
        integrator = self._integrator()
        window = integrator.for_time(0 * u.s)  # start_time < 0
        times = np.linspace(0, 1e6, 10) * u.s
        snapshots = [(t, make_blob()) for t in times]

        with pytest.raises(ValueError, match="missing before the window"):
            window.calc_sed(snapshots)

    def test_missing_states_after_the_window_raise(self):
        integrator = self._integrator()
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 2.5 * lc, 10) * u.s  # ends before end_time
        snapshots = [(t, make_blob()) for t in times]

        with pytest.raises(ValueError, match="missing after the window"):
            window.calc_sed(snapshots)

    def test_two_snapshots_exactly_at_the_window_edges(self):
        """
        The minimal legal input: one snapshot at start_time, one at end_time. This is also the
        case most sensitive to the interaction between the coverage tolerance and the clip that
        pulls the sample times back into the interpolation domain.
        """
        integrator = self._integrator()
        window = integrator.for_time(3e5 * u.s)
        snapshots = [(window.start_time, make_blob()), (window.end_time, make_blob())]

        sed = window.calc_sed(snapshots)
        # flat state, kernel integrates to 1
        assert np.isclose(sed[0].to_value(SED_UNIT), 1.0, rtol=1e-3)

class TestSedCache:
    @staticmethod
    def _counting_integrator(calls):
        def counting_sed(blob, nu):
            calls.append(id(blob))
            return np.full(len(nu), 1.0) * SED_UNIT

        return BlobLTTIntegrator(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=counting_sed
        )

    def test_each_snapshot_is_evaluated_once(self):
        calls = []
        integrator = self._counting_integrator(calls)
        lc = (1e16 * u.cm / c).to_value("s")

        times = np.linspace(0, 10 * lc, 9) * u.s
        snapshots = [(t, make_blob()) for t in times]

        # two overlapping windows over the same growing snapshot list
        integrator.for_time(2 * lc * u.s).calc_sed(snapshots[:6])
        assert len(calls) == 6
        integrator.for_time(5 * lc * u.s).calc_sed(snapshots)
        # only the three new snapshots are evaluated
        assert len(calls) == 9

    def test_replaced_snapshot_is_recomputed(self):
        calls = []
        integrator = self._counting_integrator(calls)
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)

        times = np.linspace(0, 5 * lc, 6) * u.s
        snapshots = [(t, make_blob()) for t in times]
        window.calc_sed(snapshots)
        assert len(calls) == 6

        # same time, but one state replaced by a different object -> must be re-evaluated
        snapshots[3] = (snapshots[3][0], make_blob())
        window.calc_sed(snapshots)
        assert len(calls) == 7


class TestValidation:
    def _window(self):
        integrator = BlobLTTIntegrator(
            1e16 * u.cm, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0)
        )
        return integrator.for_time(0 * u.s)

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_snapshots(self, n):
        """Interpolation needs two points; one cannot bracket a window anyway."""
        snapshots = [(t * u.s, make_blob()) for t in range(n)]
        with pytest.raises(ValueError, match="at least two"):
            self._window().calc_sed(snapshots)

    @pytest.mark.parametrize(
        "times",
        [
            pytest.param([0, 2, 1], id="non_monotonic"),
            pytest.param([0, 1, 1], id="duplicate"),
        ],
    )
    def test_non_increasing_times_are_rejected(self, times):
        snapshots = [(t, make_blob()) for t in times * u.s]
        with pytest.raises(ValueError, match="strictly increasing"):
            self._window().calc_sed(snapshots)

    def test_non_scalar_snapshot_time(self):
        """A snapshot time must be a single instant, not an array."""
        snapshots = [(0 * u.s, make_blob()), (np.array([1.0, 2.0]) * u.s, make_blob())]
        with pytest.raises(ValueError, match="must be a scalar Quantity"):
            self._window().calc_sed(snapshots)

    @pytest.mark.parametrize("n_times", [1, 3, 50])
    def test_for_time_rejects_a_time_array(self, n_times):
        """
        A time array would be broadcast against the kernel grid. With as many times as kernel
        points (the default 50) that silently yields a garbage window, so it must be rejected.
        """
        integrator = BlobLTTIntegrator(1e16 * u.cm, [1e15] * u.Hz)
        assert integrator.kernel_points_size == 50
        with pytest.raises(ValueError, match="scalar time"):
            integrator.for_time(np.linspace(0, 1e6, n_times) * u.s)

    def test_non_positive_radius(self):
        with pytest.raises(ValueError, match="positive finite length"):
            BlobLTTIntegrator(0 * u.cm, [1e15] * u.Hz)

    @pytest.mark.parametrize("R", [np.array([1e16]) * u.cm, np.array([1e16, 2e16]) * u.cm])
    def test_non_scalar_radius(self, R):
        """A shape-(1,) radius used to slip through, silently and with a numpy deprecation."""
        with pytest.raises(ValueError, match="scalar length"):
            BlobLTTIntegrator(R, [1e15] * u.Hz)

    def test_too_few_kernel_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            BlobLTTIntegrator(1e16 * u.cm, [1e15] * u.Hz, kernel_points_size=1)


class TestDocumentedWorkflow:
    """
    The module docstring's usage pattern, driven by a real TimeEvolution using its `t0` offset
    so the callback reports times on the same clock as `for_time`.
    """

    R = 1e15 * u.cm

    def _run(self):
        integrator = BlobLTTIntegrator(self.R, [1e15] * u.Hz, sed_flux_fn=flat_sed(1.0))
        window = integrator.for_time(3 * (self.R / c).to("s"))
        start, end = window.start_time, window.end_time

        blob = make_blob(self.R)
        # advance the blob to the start of the window first
        TimeEvolution(blob, total_duration_time=start,
                      energy_change_functions=synchrotron_loss(Synchrotron(blob)),
                      max_energy_change_per_interval=0.2).evaluate()

        blobs, times = ([deepcopy(blob)], [start])

        def callback(result):
            blobs.append(deepcopy(blob))
            times.append(result.blob_time)

        TimeEvolution(blob, total_duration_time=(end - start), t0=start,
                               energy_change_functions=synchrotron_loss(Synchrotron(blob)),
                               max_energy_change_per_interval=0.2,
                               distribution_change_callback=callback).evaluate()

        times = u.Quantity(times)
        return window, list(zip(times, blobs)), times

    def test_workflow_produces_a_covered_window(self):
        window, snapshots, times = self._run()

        # t0 aligned the clocks: the callback times already bracket the window
        assert len(snapshots) >= 2
        assert times[0] <= window.start_time
        assert times[-1] >= window.end_time - 1e-6 * u.s

        sed = window.calc_sed(snapshots)
        assert np.all(np.isfinite(sed.to_value(SED_UNIT)))
        assert np.isclose(sed[0].to_value(SED_UNIT), 1.0, rtol=1e-3)

