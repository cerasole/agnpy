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
:meth:`~agnpy.emission_regions.Blob.lab_time_to_blob_time`. Nothing here needs z or delta_D: the
SED evaluation reads them from the blob itself.

Usage
-----
    # build the integrator:
    integrator = BlobLTTIntegrator(blob.R_b)

    # then in the loop, for every data point that you want to calculate SED for, do:
    # find its timespan window :
    integration_window = integrator.for_time(blob.lab_time_to_blob_time(t_obs))
    start = integration_window.start_time # (can be negative, but it's fine!)
    end = integration_window.end_time

    # run the simulation till end time, gathering a list of blob state snapshots over the integration_window time span:
    snapshots = [(start, deepcopy(blob))]
    def callback(result):
        snapshots.append((result.blob_time, deepcopy(blob)))
    TimeEvolution(blob, total_duration_time=(end-start), t0=start,
                  distribution_change_callback=callback).evaluate()

    # finally, ask for the smeared SED, passing the (time, blob) snapshots and the frequencies:
    sed = integration_window.calc_sed(snapshots, nu)
    # ... and proceed with a loop

For the common case, when you know the list of time points for which you want to evaluate the SEDs, you can use a helper
function `calc_seds_over_time`, which already implements this workflow:

    seds = calc_seds_over_time(
        blob, # initial blob state
        times, # a sorted list of times for which you need SEDs
        nu_obs, # energy points for SED
        energy_change_functions=synchrotron_loss(Synchrotron(blob)) # any params needed for TimeEvolution constructor
    )

Notes:

The snapshots must bracket the window: the first one at or before
:attr:`BlobLTTWindow.start_time`, the last at or after :attr:`BlobLTTWindow.end_time`, so that
every point of the integral is reached by interpolation and never by extrapolation.
:meth:`BlobLTTWindow.calc_sed` refuses to integrate otherwise, rather than quietly returning a
flux that is too faint.

The times passed to :meth:`BlobLTTWindow.calc_sed` must use the same t=0 reference point as times
passed to :meth:`BlobLTTIntegrator.for_time`. Pass ``t0`` to the ``TimeEvolution`` constructor
to align the two: it shifts the time reported to the callback onto the same clock, so no manual
offsetting is needed.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import astropy.units as u
from astropy.constants import c as c_light
from scipy.interpolate import interp1d

from agnpy import Blob
from agnpy.time_evolution.time_evolution import TimeEvolution

__all__ = [
    "BlobLTTIntegrator",
    "BlobLTTWindow",
    "calc_seds_over_time",
]

# Speed of light in CGS. All internal arrays are plain floats in CGS; Quantity is used only at
# the API boundary.
_C_CGS = c_light.to("cm/s").value

_SED_UNIT = u.Unit("erg / (cm2 s)")

# Radii closer than this (relatively) are considered consistent when validating snapshots.
_RADIUS_RTOL = 1e-6

# Relative slack when checking that snapshots cover the window, to absorb float noise in times.
_COVERAGE_RTOL = 1e-14


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


def _default_sed_flux(blob: Blob, nu: u.Quantity) -> u.Quantity:
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
    # blob-frame times of the quadrature points, i.e. the requested time plus the kernel offsets
    _quadrature_times_s: np.ndarray
    _W_cgs: np.ndarray

    @property
    def start_time(self) -> u.Quantity:
        """
        Earliest blob-frame time contributing, emitted by the far side of the blob: t0 - R/c.

        May be negative -- ``for_time(0)`` is the intended way to discover how much blob state
        is needed before the nominal start of a run.
        """
        return self._quadrature_times_s[0] * u.s

    @property
    def end_time(self) -> u.Quantity:
        """
        Latest blob-frame time contributing, emitted by the near side of the blob: t0 + R/c.

        The simulation must be advanced at least this far before the SED can be computed.
        """
        return self._quadrature_times_s[-1] * u.s

    def calc_sed(
        self,
        snapshots: Sequence[Tuple[u.Quantity, Blob]],
        nu_obs: u.Quantity,
    ) -> u.Quantity:
        """
        Integrate the blob states over the window to get the observed, smeared SED.

        The snapshots must bracket the window -- the earliest time at or before
        :attr:`start_time` and the latest at or after :attr:`end_time` -- so that the whole
        integral is covered by interpolation and none of it by extrapolation. At least two
        snapshots are therefore required.

        Parameters
        ----------
        snapshots : sequence of (time, blob)
            Each pair is a blob-frame time and a snapshot of the blob at that time. At least
            two pairs, sorted by strictly increasing time, on the same clock as the time passed
            to :meth:`BlobLTTIntegrator.for_time`. Each blob must be an independent snapshot:
            ``TimeEvolution`` mutates one blob in place, so appending the live object repeatedly
            yields N references to a single final state.
        nu_obs : :class:`~astropy.units.Quantity`
            Observed frequencies to evaluate the SED at.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Flux at each frequency of ``nu_obs``, in erg / (cm2 s).

        Raises
        ------
        ValueError
            If fewer than two snapshots are given, the times are not strictly increasing, a
            snapshot's ``R_b`` disagrees with the integrator's radius, or the snapshots do not
            bracket the window.
        """
        integrator = self._integrator
        nu_hz = nu_obs.to("Hz")

        if len(snapshots) < 2:
            raise ValueError(
                f"at least two blob snapshots are required to interpolate over the window, "
                f"got {len(snapshots)}"
            )

        snapshot_times_s = np.empty(len(snapshots), dtype=float)
        snapshots_s = []
        for i, (t, blob) in enumerate(snapshots):
            if not t.isscalar:
                raise ValueError(
                    f"snapshot {i}: time must be a scalar Quantity, got shape {t.shape}"
                )
            snapshot_times_s[i] = t.to_value("s")
            snapshots_s.append((snapshot_times_s[i], blob))

        if not np.all(np.diff(snapshot_times_s) > 0):
            raise ValueError("snapshot times must be strictly increasing")

        integrator._validate_radii(snapshots_s)
        self._validate_coverage(snapshot_times_s)

        table = integrator._sed_table(snapshots_s, nu_hz)  # (n_nu, n_snapshots)

        quadrature_times_s = self._quadrature_times_s
        # The snapshots bracket the window, so this only pulls the endpoints back inside the
        # interpolation domain by the tolerance _validate_coverage allows. The integral itself
        # still runs over the true grid, whose spacing the clip must not disturb.
        clipped_quadrature_times_s = np.clip(
            quadrature_times_s, snapshot_times_s[0], snapshot_times_s[-1]
        )

        # No fill_value: the clip above guarantees the domain, so an out-of-bounds sample would
        # be a bug and should raise rather than silently contribute zero.
        interp = interp1d(
            snapshot_times_s, table, axis=1, kind="linear", assume_sorted=True
        )
        seds = interp(clipped_quadrature_times_s)  # (n_nu, n_kernel)
        return np.trapz(
            self._W_cgs[np.newaxis, :] * seds, quadrature_times_s, axis=1
        ) << _SED_UNIT

    def _validate_coverage(self, snapshot_times_s: np.ndarray) -> None:
        """
        Require the snapshots to bracket the window, so that every point of the integral is
        reached by interpolation rather than extrapolation.
        """
        start_s = self._quadrature_times_s[0]
        end_s = self._quadrature_times_s[-1]
        span = max(end_s - start_s, abs(end_s), 1.0)
        slack = _COVERAGE_RTOL * span

        if snapshot_times_s[0] > start_s + slack:
            raise ValueError(
                f"blob states are missing before the window: it starts at {start_s:.6g} s but "
                f"the earliest snapshot is at {snapshot_times_s[0]:.6g} s, a gap of "
                f"{snapshot_times_s[0] - start_s:.6g} s. Start the simulation "
                f"{snapshot_times_s[0] - start_s:.6g} s earlier (see BlobLTTWindow.start_time)."
            )
        if snapshot_times_s[-1] < end_s - slack:
            raise ValueError(
                f"blob states are missing after the window: it ends at {end_s:.6g} s but the "
                f"latest snapshot is at {snapshot_times_s[-1]:.6g} s, a gap of "
                f"{end_s - snapshot_times_s[-1]:.6g} s. Advance the simulation to at least "
                f"{end_s:.6g} s (see BlobLTTWindow.end_time)."
            )


class BlobLTTIntegrator:
    """
    Light-travel-time integrator for a spherical blob of constant radius.

    SEDs of individual snapshots are cached, so repeated ``for_time(...).calc_sed(...)`` calls
    over an overlapping set of snapshots evaluate each snapshot only once. Cache entries for
    snapshots no longer covered by the time window are discarded.

    Parameters
    ----------
    R : :class:`~astropy.units.Quantity`
        Blob radius in the blob frame; must be a scalar length.
    kernel_points_size : int
        Number of quadrature points across the blob diameter. The default gives roughly 1e-3
        relative accuracy; the quadrature is second order, so doubling it cuts the error by
        about four. Raising it is cheap, as the kernel is computed once.
    sed_flux_fn : callable, optional
        ``f(blob, nu) -> Quantity[erg / (cm2 s)]``. Defaults to Synchrotron + SSC; override to
        add external Compton or absorption.
    """

    def __init__(self, R: u.Quantity, *,
                 kernel_points_size: int = 50, sed_flux_fn=None):
        if not R.isscalar:
            raise ValueError(f"blob radius must be a scalar length, got shape {R.shape}")
        R_cm = R.to("cm").value
        if not np.isfinite(R_cm) or R_cm <= 0:
            raise ValueError(f"blob radius must be a positive finite length, got {R}")
        if kernel_points_size < 2:
            raise ValueError(
                f"kernel_points_size must be at least 2, got {kernel_points_size}"
            )

        self._R = R.to("cm")
        self._R_cm = R_cm
        self._sed_flux_fn = sed_flux_fn if sed_flux_fn is not None else _default_sed_flux
        # R and the sampling are fixed, so the kernel never changes: compute it once.
        self._tau_s, self._W_cgs = _constant_kernel_cgs(R_cm, kernel_points_size)
        # (snapshot time [s], nu_obs bytes) -> (id(blob), sed row); see _sed_table.
        self._sed_cache: dict[tuple[float, bytes], tuple[int, np.ndarray]] = {}

    @property
    def R(self) -> u.Quantity:
        """
        The blob radius this integrator assumes.

        Each snapshot's ``R_b`` is checked against this value by
        :meth:`BlobLTTWindow.calc_sed`.
        """
        return self._R

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
        if not t_blob.isscalar:
            raise ValueError(
                f"t_blob must be a scalar time, got shape {t_blob.shape}. Call for_time once "
                "per time; a time array would be broadcast against the kernel grid."
            )
        return BlobLTTWindow(self, t_blob.to("s").value + self._tau_s, self._W_cgs)

    def _validate_radii(self, snapshots_s: Sequence[Tuple[float, Blob]]) -> None:
        for i, (t, blob) in enumerate(snapshots_s):
            actual = blob.R_b.to("cm").value
            if not np.isclose(actual, self._R_cm, rtol=_RADIUS_RTOL, atol=0.0):
                raise ValueError(
                    f"snapshot {i} at t = {t:.6g} s has R_b = {actual:.6e} cm "
                    f"but the integrator was built for R = {self._R_cm:.6e} cm. All snapshots "
                    "must share the integrator's radius."
                )

    def _sed_table(
        self, snapshots_s: Sequence[Tuple[float, Blob]], nu_hz: u.Quantity
    ) -> np.ndarray:
        """
        SED of every snapshot as an (n_nu, n_snapshots) array, reusing cached rows.

        A cached row is reused only when the blob at that time is the same object *and* it was
        computed for this same ``nu_hz`` grid.
        """
        cache = self._sed_cache
        nu_key = nu_hz.to_value("Hz").tobytes()
        table = np.empty((nu_hz.size, len(snapshots_s)), dtype=float)
        fresh: dict[tuple[float, bytes], tuple[int, np.ndarray]] = {}

        for i, (t, blob) in enumerate(snapshots_s):
            key = (float(t), nu_key)
            cached = cache.get(key)
            if cached is not None and cached[0] == id(blob):
                row = cached[1]
            else:
                row = np.asarray(
                    self._sed_flux_fn(blob, nu_hz).to_value(_SED_UNIT), dtype=float
                )
            table[:, i] = row
            fresh[key] = (id(blob), row)

        self._sed_cache = fresh
        return table


# Constructor arguments calc_seds_over_time manages itself and refuses to accept from the caller.
_RESERVED_TIME_EVOLUTION_KWARGS = frozenset(
    {"blob", "total_duration_time", "t0", "distribution_change_callback"}
)

def calc_seds_over_time(
    blob: Blob,
    times: u.Quantity,
    nu_obs: u.Quantity,
    *,
    kernel_points_size: int = 50,
    sed_flux_fn=None,
    assume_steady_before_start: bool = True,
    **time_evolution_kwargs,
) -> list[u.Quantity]:
    """
   Utility function to run TimeEvolution starting from time = 0 over ``times`` and return the light-travel-time smeared SED at each one.

    Parameters
    ----------
    blob : :class:`~agnpy.emission_regions.Blob`
        The blob in its state at blob-frame time 0. Mutated in place.
    times : :class:`~astropy.units.Quantity`
        Blob-frame times to compute the SED at, strictly increasing.
    nu_obs : :class:`~astropy.units.Quantity`
        Observed frequencies the SEDs are evaluated at; forwarded to :class:`BlobLTTIntegrator`.
    kernel_points_size, sed_flux_fn
        Forwarded to :class:`BlobLTTIntegrator`.
    assume_steady_before_start : bool
        The first requested time's window may reach before blob-frame time 0, if its
        light-crossing margin is larger than ``times[0]`` itself. When ``True`` (the default),
        ``blob``'s given state is treated as unchanged for as far back that window needs --
        the same assumption the manual workflow above makes implicitly by seeding at
        ``start_time`` rather than at 0. When ``False``, that situation raises instead, naming
        how much earlier ``blob``'s history would need to start.
    **time_evolution_kwargs
        Forwarded to every internal :class:`~agnpy.time_evolution.TimeEvolution` call, e.g.
        ``energy_change_functions``, ``max_energy_change_per_interval``, ``method``. Must not
        include ``blob``, ``total_duration_time``, ``t0`` or ``distribution_change_callback``,
        which this function manages itself. Passing ``expansion`` is not meaningful here: this
        integrator assumes a constant radius, so an expanding blob's later snapshots will fail
        the radius check in :meth:`BlobLTTWindow.calc_sed`.

    Returns
    -------
    list of :class:`~astropy.units.Quantity`
        One SED per entry in ``times``, each of shape ``nu_obs.shape``, in erg / (cm2 s).
    """
    conflicting = _RESERVED_TIME_EVOLUTION_KWARGS & time_evolution_kwargs.keys()
    if conflicting:
        raise ValueError(
            f"calc_seds_over_time manages {sorted(conflicting)} itself; do not pass "
            "them in time_evolution_kwargs"
        )

    times = u.Quantity(times)
    if times.isscalar:
        times = times.reshape(1)
    if len(times) == 0:
        return []
    if len(times) > 1 and not np.all(np.diff(times.to_value("s")) > 0):
        # We could sort them here, but it's probably better to let the user do it
        # (passing unsorted times might be a sign of a user mistake, so if we sort them, we will mask the problem)
        raise ValueError("times must be strictly increasing")

    integrator = BlobLTTIntegrator(
        blob.R_b, kernel_points_size=kernel_points_size, sed_flux_fn=sed_flux_fn
    )

    snapshots = []
    first_start = integrator.for_time(times[0]).start_time
    initial_state_snapshot = deepcopy(blob)
    now = 0 * u.s
    if first_start < now:
        if assume_steady_before_start:
            snapshots.append((first_start, initial_state_snapshot))
        else:
            raise ValueError(
                f"The window for the first requested time starts at {first_start}, before "
                f"blob-frame time 0. Give blob a history starting {-first_start} earlier, or pass "
                "assume_steady_before_start=True to treat its given state as unchanged that far back."
            )

    snapshots.append((now, initial_state_snapshot))

    def callback(result):
        snapshots.append((result.blob_time, deepcopy(blob)))

    seds = []
    for t in times:
        window = integrator.for_time(t)

        if window.start_time > now:
            # just fast-forward to the start of the window
            TimeEvolution(
                blob, total_duration_time=(window.start_time - now),
                **time_evolution_kwargs,
            ).evaluate()
            now = window.start_time
            # take a snapshot at the start of the window
            snapshots.append((now, deepcopy(blob)))

        if window.end_time > now:
            # proceed till the end of the window, gathering snapshots on the way
            TimeEvolution(
                blob, total_duration_time=(window.end_time - now), t0=now,
                distribution_change_callback=callback, **time_evolution_kwargs,
            ).evaluate()
            now = window.end_time

        # drop earlier snapshots no longer needed now
        keep_from = 0
        for i, (snap_t, _) in enumerate(snapshots):
            if snap_t <= window.start_time:
                keep_from = i
            else:
                break
        snapshots = snapshots[keep_from:]

        # finally, calculate the actual SED and add it to the final list
        sed = window.calc_sed(snapshots, nu_obs)
        seds.append(sed)

    return seds
