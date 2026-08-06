"""
Light-Travel-Time (LTT) integration for time-variable AGN blob SEDs.

Problem
-------
A blob does not radiate as a single snapshot. Photons that reach the observer together were
emitted at different blob-frame times from different depths along the line of sight: the volume
element at line-of-sight offset xi (positive = towards the observer) contributes emission from
blob-frame time t0 + xi/c. Writing tau = xi/c, the observed SED at blob-frame time t0 is

    F(nu, t0) = integral W(tau) * F_std(nu, t0 + tau) dtau

where F_std(nu, t') is the ordinary agnpy SED of a uniform blob in state t', and W is a purely
geometric kernel: the cross-section of the sphere at offset tau, divided by the blob volume. For
a blob of constant radius R, with V = 4/3 pi R^3,

    W(tau) = pi c (R^2 - c^2 tau^2) / V = (3c / 4R) (1 - (c tau / R)^2)

The integration limits are such that tau spans [-R/c, +R/c].
The kernel is a symmetric parabola vanishing at both ends, and

    integral W dtau = 1     exactly,

so a blob whose state does not change reproduces the ordinary SED.

Because R and the sampling points are both fixed, the kernel is computed once when
the integrator is built and reused for every requested time.

All times in this module are blob-frame times. Convert an observer-frame time with
:func:`~agnpy.utils.conversion.lab_time_to_blob_time`. Nothing here needs z or delta_D: the SED
evaluation reads them from the blob itself.

Usage
-----
The simplest usage: build the integrator, then call for_time(time).calc_sed() on it, to get the SED:

    from agnpy.time_evolution import ltt_integrator_constant_radius
    from agnpy.utils.conversion import lab_time_to_blob_time

    integrator = ltt_integrator_constant_radius(blob.R_b, nu_obs=nu)

    # run the time evaluation of the blob, caching blob-copy snapshots together with corresponding times
    snapshots, snap_times = [], []
    def callback(result):
        snapshots.append(deepcopy(blob))
        snap_times.append(result.total_time)
    TimeEvolution(blob, total_blob_simulation_time, ...,
                  distribution_change_callback=callback).evaluate()

    # 3. ask for the smeared SED:
    sed_blob_time = lab_time_to_blob_time(sed_lab_time, z=blob.z, delta_D=blob.delta_D)
    sed = integrator.for_time(sed_blob_time).calc_sed(snapshots, snap_times)

The above example assumes that:

    0 + tau < sed_blob_time < total_blob_simulation_time - tau

It means that, in order to calculate the smeared SED at a given time t, you must have a time margin [-tau, +tau]
of time-evaluated blob data. For this reason, the object returned by `BlobLTTIntegrator.for_time` has additional
`start_time` and `end_time` properties, which represent central_time +/- tau,
which can help the caller know how far to run the simulation and which snapshots may be dropped:

    integrator = ltt_integrator_constant_radius(blob.R_b, nu_obs=nu)

    # 1. Suppose we want the SED at t=0 (note: all times in this example are in the blob frame)
    start_window = integrator.for_time(0 * u.s)
    lead = -start_window.start_time # will be negative, so multiply by -1 to get absolute value of the additional simulation time needed
    tail = start_window.end_time

    # 2. start the run `lead` earlier, and run until "lead+tail" time
    snapshots, snap_times = [], []
    # ... run the time evolution, recording blobs and corresponding times in these arrays, and then:
    sed = start_window.calc_sed(snapshots, snap_times)

    # 3. Now again if you want to find SED at further time, repeat the same procedure, but remember about `lead` shift:
    end_window = integrator.for_time(lead + end_time)
    # run simulation until end_window.end_time, caching all blobs generated in the [end_window.start_time, end_window.end_time] range
    sed = end_window.calc_sed(snapshots, snap_times)


Warning:

    The times passed to :meth:`BlobLTTWindow.calc_sed` must use the same t=0 reference point as times
    passed to :meth:`BlobLTTIntegrator.for_time`. ``TimeEvolution`` always reports elapsed time
    from zero, so a run started ``lead`` early must subtract ``lead`` when recording snapshot
    times. Getting this wrong shifts every SED by ``lead`` and ``calc_sed`` will
    usually not detect it, because the window still looks covered.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import astropy.units as u
from astropy.constants import c as c_light
from scipy.interpolate import interp1d

__all__ = [
    "BlobLTTIntegrator",
    "BlobLTTWindow",
    "ltt_integrator_constant_radius",
]

# Speed of light in CGS. All internal arrays are plain floats in CGS; Quantity is used only at
# the API boundary.
_C_CGS = c_light.to("cm/s").value

_SED_UNIT = u.Unit("erg / (cm2 s)")

# Radii closer than this (relatively) are considered consistent when validating snapshots.
_RADIUS_RTOL = 1e-6

# Relative slack when checking that snapshots cover the window, to absorb float noise in times.
_COVERAGE_RTOL = 1e-9


def _constant_kernel_cgs(R_cm: float, n_points: int):
    """
    Geometric LTT kernel for a blob of constant radius, in CGS floats.

    Returns (tau_s, W_cgs): offsets [s] spanning [-R/c, +R/c] and weights [1/s] forming the
    symmetric parabola (3c / 4R)(1 - (c tau / R)^2), normalised so that the integral is 1.
    """
    rho = np.linspace(-1.0, 1.0, n_points)
    tau_s = rho * R_cm / _C_CGS
    W_cgs = 0.75 * (_C_CGS / R_cm) * (1.0 - rho ** 2)
    return tau_s, W_cgs


def _default_sed_flux(blob, nu: u.Quantity) -> u.Quantity:
    """Synchrotron + SSC flux, the usual single-zone SED."""
    from agnpy import Synchrotron, SynchrotronSelfCompton

    return Synchrotron(blob).sed_flux(nu) + SynchrotronSelfCompton(blob).sed_flux(nu)


@dataclass(frozen=True)
class BlobLTTWindow:
    """
    The span of blob states needed to compute one light-travel-time smeared SED.

    All times are BLOB-FRAME times, on the same clock as the time passed to
    :meth:`BlobLTTIntegrator.for_time`. Obtained from that method rather than constructed
    directly.
    """

    _integrator: "BlobLTTIntegrator"
    _t0_s: float
    _tau_s: np.ndarray
    _W_cgs: np.ndarray

    @property
    def central_time(self) -> u.Quantity:
        """Blob-frame time this SED is for; equal to the time passed to ``for_time``."""
        return self._t0_s * u.s

    @property
    def start_time(self) -> u.Quantity:
        """
        Earliest blob-frame time contributing, emitted by the far side of the blob: t0 - R/c.

        May be negative -- ``for_time(0)`` is the intended way to discover how much blob state
        is needed before the nominal start of a run.
        """
        return (self._t0_s + self._tau_s[0]) * u.s

    @property
    def end_time(self) -> u.Quantity:
        """
        Latest blob-frame time contributing, emitted by the near side of the blob: t0 + R/c.

        The simulation must be advanced at least this far before the SED can be computed.
        """
        return (self._t0_s + self._tau_s[-1]) * u.s

    @property
    def kernel(self):
        """
        The quadrature grid as ``(tau, W)`` Quantities: offsets from :attr:`central_time` and
        the geometric weights. Useful for plotting or diagnostics; ``calc_sed`` uses it
        internally.
        """
        return self._tau_s * u.s, self._W_cgs / u.s

    def calc_sed(
        self,
        blobs: Sequence,
        times: u.Quantity,
        *,
        no_state_before_first: bool = False,
    ) -> u.Quantity:
        """
        Integrate the blob states over the window to get the observed, smeared SED.

        Parameters
        ----------
        blobs : sequence of :class:`~agnpy.emission_regions.Blob`
            Snapshots of the blob. These must be independent snapshots: ``TimeEvolution``
            mutates one blob in place, so appending the live object repeatedly yields N
            references to a single final state.
        times : :class:`~astropy.units.Quantity`
            Blob-frame time of each snapshot, strictly increasing, same length as ``blobs``,
            on the same clock as :attr:`central_time`.
        no_state_before_first : bool
            By default a window reaching before ``times[0]`` is an error, because the flux
            there is simply unknown -- run the simulation earlier (see :attr:`start_time`).
            Set this to ``True`` to assert instead that the blob did not emit at all before
            the first snapshot, in which case that part of the window contributes zero.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Flux at each frequency of the integrator's ``nu_obs``, in erg / (cm2 s).

        Raises
        ------
        ValueError
            If ``blobs`` and ``times`` disagree in length, ``times`` is not strictly
            increasing, a snapshot's ``R_b`` disagrees with the integrator's radius, or the
            snapshots do not cover the window.
        """
        integrator = self._integrator
        t_s = np.atleast_1d(times.to("s").value).astype(float)

        if len(blobs) != t_s.size:
            raise ValueError(
                f"blobs and times must have the same length, got {len(blobs)} and {t_s.size}"
            )
        if t_s.size == 0:
            raise ValueError("at least one blob snapshot is required")
        if t_s.size > 1 and not np.all(np.diff(t_s) > 0):
            raise ValueError("times must be strictly increasing")

        integrator._validate_radii(blobs, t_s)
        self._validate_coverage(t_s, no_state_before_first)

        table = integrator._sed_table(blobs, t_s)  # (n_nu, n_t)

        tau_s, W_cgs = self._tau_s, self._W_cgs
        t_sample = self._t0_s + tau_s

        if t_s.size == 1:
            # Frozen state across the whole window; no interpolation needed.
            if no_state_before_first:
                W_cgs = np.where(t_sample < t_s[0], 0.0, W_cgs)
            return (table[:, 0] * np.trapz(W_cgs, tau_s)) << _SED_UNIT

        if no_state_before_first:
            # Clip only the top; interp1d zero-fills below times[0].
            t_sample = np.minimum(t_sample, t_s[-1])
        else:
            # Coverage is already established; this only absorbs float noise at the edges.
            t_sample = np.clip(t_sample, t_s[0], t_s[-1])

        interp = interp1d(
            t_s, table, axis=1, kind="linear",
            bounds_error=False, fill_value=0.0, assume_sorted=True,
        )
        seds = interp(t_sample)  # (n_nu, n_kernel)
        return np.trapz(W_cgs[np.newaxis, :] * seds, tau_s, axis=1) << _SED_UNIT

    def _validate_coverage(self, t_s: np.ndarray, no_state_before_first: bool) -> None:
        start_s = self._t0_s + self._tau_s[0]
        end_s = self._t0_s + self._tau_s[-1]
        span = max(end_s - start_s, abs(end_s), 1.0)
        slack = _COVERAGE_RTOL * span

        if not no_state_before_first and start_s < t_s[0] - slack:
            raise ValueError(
                f"blob states are missing before the window: it starts at {start_s:.6g} s but "
                f"the earliest snapshot is at {t_s[0]:.6g} s, a gap of {t_s[0] - start_s:.6g} s. "
                f"Start the simulation {t_s[0] - start_s:.6g} s earlier (see "
                "BlobLTTWindow.start_time), or pass no_state_before_first=True if the blob "
                "genuinely did not emit before the first snapshot."
            )
        if end_s > t_s[-1] + slack:
            raise ValueError(
                f"blob states are missing after the window: it ends at {end_s:.6g} s but the "
                f"latest snapshot is at {t_s[-1]:.6g} s, a gap of {end_s - t_s[-1]:.6g} s. "
                f"Advance the simulation to at least {end_s:.6g} s (see BlobLTTWindow.end_time)."
            )


class BlobLTTIntegrator:
    """
    Light-travel-time integrator for a spherical blob of constant radius.

    Construct with :func:`ltt_integrator_constant_radius` rather than directly.

    SEDs of individual snapshots are cached, so repeated ``for_time(...).calc_sed(...)`` calls
    over an overlapping set of snapshots evaluate each snapshot only once. Cache entries for
    snapshots no longer passed in are discarded, so the cache never keeps blobs alive.
    """

    def __init__(self, R: u.Quantity, nu_obs: u.Quantity, n_points: int, sed_flux_fn=None):
        R_cm = float(R.to("cm").value)
        if not np.isfinite(R_cm) or R_cm <= 0:
            raise ValueError(f"blob radius must be a positive finite length, got {R}")
        if n_points < 2:
            raise ValueError(f"kernel_points_size must be at least 2, got {n_points}")

        self._R = R.to("cm")
        self._R_cm = R_cm
        self._nu_obs = nu_obs
        self._nu_hz = nu_obs.to("Hz")
        self._sed_flux_fn = sed_flux_fn if sed_flux_fn is not None else _default_sed_flux
        # R and n_points are fixed, so the kernel never changes: compute it once.
        self._tau_s, self._W_cgs = _constant_kernel_cgs(R_cm, n_points)
        # t_s -> (id(blob), sed row); see _sed_table.
        self._sed_cache: dict[float, tuple[int, np.ndarray]] = {}

    @property
    def R(self) -> u.Quantity:
        """
        The blob radius this integrator assumes.

        Each snapshot's ``R_b`` is checked against this value by
        :meth:`BlobLTTWindow.calc_sed`.
        """
        return self._R

    @property
    def nu_obs(self) -> u.Quantity:
        """Frequencies the SEDs are evaluated at, as supplied."""
        return self._nu_obs

    @property
    def kernel_points_size(self) -> int:
        """Number of quadrature points across the blob."""
        return self._tau_s.size

    def for_time(self, t_blob: u.Quantity) -> BlobLTTWindow:
        """
        Describe the blob states needed for the SED at blob-frame time ``t_blob``.

        This is a purely geometric query: it never inspects snapshots and never fails on
        coverage grounds. ``for_time(0)`` legitimately returns a negative
        :attr:`~BlobLTTWindow.start_time`, which is how you discover how much blob state is
        needed before the nominal start of a run.
        """
        t0_s = float(t_blob.to("s").value)
        return BlobLTTWindow(self, t0_s, self._tau_s, self._W_cgs)

    def _validate_radii(self, blobs: Sequence, t_s: np.ndarray) -> None:
        for i, blob in enumerate(blobs):
            actual = blob.R_b.to("cm").value
            if not np.isclose(actual, self._R_cm, rtol=_RADIUS_RTOL, atol=0.0):
                raise ValueError(
                    f"snapshot {i} at t = {t_s[i]:.6g} s has R_b = {actual:.6e} cm but the "
                    f"integrator was built for R = {self._R_cm:.6e} cm. All snapshots must "
                    "share the integrator's radius."
                )

    def _sed_table(self, blobs: Sequence, t_s: np.ndarray) -> np.ndarray:
        """
        SED of every snapshot as an (n_nu, n_t) array, reusing cached rows.

        A cached row is reused only when the blob at that time is the same object, so a
        recomputed or replaced snapshot is re-evaluated. Afterwards the cache is pruned to the
        times passed in, which both keeps it bounded and guarantees every cached id belongs to
        a blob the caller still holds -- so ids cannot have been recycled behind our back.
        """
        cache = self._sed_cache
        table = np.empty((self._nu_hz.size, t_s.size), dtype=float)
        fresh: dict[float, tuple[int, np.ndarray]] = {}

        for i, (blob, t) in enumerate(zip(blobs, t_s)):
            key = float(t)
            cached = cache.get(key)
            if cached is not None and cached[0] == id(blob):
                row = cached[1]
            else:
                row = np.asarray(
                    self._sed_flux_fn(blob, self._nu_hz).to_value(_SED_UNIT), dtype=float
                )
            table[:, i] = row
            fresh[key] = (id(blob), row)

        self._sed_cache = fresh
        return table


def ltt_integrator_constant_radius(
    R: u.Quantity,
    nu_obs: u.Quantity,
    *,
    kernel_points_size: int = 50,
    sed_flux_fn=None,
) -> BlobLTTIntegrator:
    """
    LTT integrator for a blob of constant radius.

    Parameters
    ----------
    R : :class:`~astropy.units.Quantity`
        Blob radius in the blob frame.
    nu_obs : :class:`~astropy.units.Quantity`
        Observed frequencies the SEDs are evaluated at.
    kernel_points_size : int
        Number of quadrature points across the blob diameter. The default gives roughly 1e-3
        relative accuracy; the quadrature is second order, so doubling it cuts the error by
        about four. Raising it is cheap, as the kernel is computed once.
    sed_flux_fn : callable, optional
        ``f(blob, nu) -> Quantity[erg / (cm2 s)]``. Defaults to Synchrotron + SSC; override to
        add external Compton or absorption.

    Returns
    -------
    :class:`BlobLTTIntegrator`
    """
    return BlobLTTIntegrator(R, nu_obs, kernel_points_size, sed_flux_fn)
