import numpy as np
from astropy.constants import h, c, m_e, sigma_T
import astropy.units as u
from .compton import cos_psi, x_re_shell, x_re_ring, mu_star
from .targets import SSDisk, SphericalShellBLR, RingDustTorus


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
# equivalency to transform frequencies to energies in electron rest mass units
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]


__all__ = ["sigma", "Absorption"]


def sigma(s):
    """photon-photon pair production cross section, Eq. 17 of [Dermer2009]"""
    beta_cm = np.sqrt(1 - np.power(s, -1))
    _prefactor = 3 / 16 * sigma_T * (1 - np.power(beta_cm, 2))
    _term1 = (3 - np.power(beta_cm, 4)) * np.log((1 + beta_cm) / (1 - beta_cm))
    _term2 = -2 * beta_cm * (2 - np.power(beta_cm, 2))
    values = _prefactor * (_term1 + _term2)
    values[s < 1] = 0
    return values


class Absorption:
    """class to compute the absorption due to gamma-gamma pair production

    Parameters
    ----------
    blob : :class:`~agnpy.emission_regions.Blob`
        emission region and electron distribution hitting the photon target
    target : :class:`~agnpy.targets`
        class describing the target photon field
    r : :class:`~astropy.units.Quantity`
        distance of the blob from the Black Hole (i.e. from the target photons)
    """

    def __init__(self, blob, target, r):
        self.blob = blob
        self.target = target
        self.r = r
        self.set_mu()
        self.set_phi()
        self.set_l()

    def set_mu(self, mu_size=100):
        self.mu_size = mu_size
        self.mu = np.linspace(-1, 1, self.mu_size)

    def set_phi(self, phi_size=50):
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)

    def set_l(self, l_size=10):
        """set the range of integration for the distance
        """
        # integrate up 3000 pc
        self.l_size = l_size
        l_max = 3000 * u.pc
        self.l = (
            np.logspace(
                np.log10(self.r.to_value("cm")),
                np.log10(l_max.to_value("cm")),
                self.l_size,
            )
            * u.cm
        )

    def _opacity_disk(self, nu):
        """opacity generated by a Shakura Sunyaev disk

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are
            observed frequencies (observer frame).
        """
        # define the dimensionless energy
        epsilon_1 = nu.to("", equivalencies=epsilon_equivalency)
        # transform to BH frame
        epsilon_1 *= 1 + self.blob.z
        # each value of l, distance from the BH, defines a different range of
        # cosine integration, we have to break the integration as the array of
        # mu takes different values at each distance
        integral = np.empty(len(epsilon_1))
        for i, _epsilon_1 in enumerate(epsilon_1):
            integrand_l = np.empty(len(self.l))
            for j, _l in enumerate(self.l):
                print("j: ", j)
                print("l: ", _l)
                l_tilde = (_l / self.target.R_g).to_value("")
                # for the multidimensional integration on the angles only
                # axis 0 : phi
                # axis 1 : mu
                _phi = np.reshape(self.phi, (1, self.phi.size))
                mu = self.target.mu_from_r_tilde(l_tilde)
                print("mu: ", mu)
                print("-------------")
                _mu = np.reshape(mu, (mu.size, 1))
                # epsilon and phi disk have the same dimension as mu
                _epsilon = self.target.epsilon_mu(_mu, l_tilde)
                _phi_disk_mu = self.target.phi_disk_mu(_mu, l_tilde)
                _cos_psi = cos_psi(self.blob.mu_s, _mu, _phi)
                _s = _epsilon_1 * _epsilon * (1 - _cos_psi) / 2
                _integrand_mu = _phi_disk_mu / (
                    _epsilon
                    * np.power(_l, 3)
                    * _mu
                    * np.power(np.power(_mu, -2) - 1, 3 / 2)
                )
                _integrand = _integrand_mu * (1 - _cos_psi) * sigma(_s)
                # integrate over mu and phi
                integral_mu = np.trapz(_integrand, mu, axis=0)
                integral_phi = np.trapz(integral_mu, self.phi, axis=0)
                integrand_l[i] = integral_phi.to_value("cm-1")
            # integrate over l
            integral[j] = np.trapz(integrand_l, self.l, axis=0).to_value("cm")

        prefactor_num = 3 * self.target.L_disk * self.target.R_g
        prefactor_denum = 16 * np.pi * self.target.eta * m_e * np.power(c, 3)
        tau = prefactor_num / prefactor_denum * integral
        return tau

    def _opacity_shell_blr(self, nu):
        """opacity generated by a spherical shell Broad Line Region

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are
            observed frequencies (observer frame).
        """
        # define the dimensionless energy
        epsilon_1 = nu.to("", equivalencies=epsilon_equivalency)
        # transform to BH frame
        epsilon_1 *= 1 + self.blob.z
        # for multidimensional integration
        # axis 0: mu_re
        # axis 1: phi
        # axis 2: l
        # axis 3: epsilon_1
        # arrays starting with _ are multidimensional and used for integration
        _mu = np.reshape(self.mu, (self.mu.size, 1, 1, 1))
        _phi = np.reshape(self.phi, (1, self.phi.size, 1, 1))
        _l = np.reshape(self.l, (1, 1, self.l.size, 1))
        _epsilon_1 = np.reshape(epsilon_1, (1, 1, 1, epsilon_1.size))
        # define integrating function
        _x = x_re_shell(_mu, self.target.R_line, _l)
        _mu_star = mu_star(_mu, self.target.R_line, _l)

        _cos_psi = cos_psi(self.blob.mu_s, _mu_star, _phi)
        _s = _epsilon_1 * self.target.epsilon_line * (1 - _cos_psi) / 2
        _integrand = (1 - _cos_psi) * np.power(_x, -2) * sigma(_s)

        prefactor_num = self.target.xi_line * self.target.L_disk
        prefactor_denum = (
            np.power(4 * np.pi, 2) * self.target.epsilon_line * m_e * np.power(c, 3)
        )

        integral_mu = np.trapz(_integrand, self.mu, axis=0)
        integral_phi = np.trapz(integral_mu, self.phi, axis=0)
        integral = np.trapz(integral_phi, self.l, axis=0)

        tau = prefactor_num / prefactor_denum * integral
        return tau.to_value("")

    def _opacity_ring_torus(self, nu):
        """opacity generated by a ring Dust Torus

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are
            observed frequencies (observer frame).
        """
        # define the dimensionless energy
        epsilon_1 = nu.to("", equivalencies=epsilon_equivalency)
        # transform to BH frame
        epsilon_1 *= 1 + self.blob.z
        # for multidimensional integration
        # axis 0: phi
        # axis 1: l
        # axis 2: epsilon_1
        # arrays starting with _ are multidimensional and used for integration
        _phi = np.reshape(self.phi, (self.phi.size, 1, 1))
        _l = np.reshape(self.l, (1, self.l.size, 1))
        _epsilon_1 = np.reshape(epsilon_1, (1, 1, epsilon_1.size))
        _x = x_re_ring(self.target.R_dt, _l)
        _mu = _l / _x

        _cos_psi = cos_psi(self.blob.mu_s, _mu, _phi)
        _s = _epsilon_1 * self.target.epsilon_dt * (1 - _cos_psi) / 2
        _integrand = (1 - _cos_psi) * np.power(_x, -2) * sigma(_s)

        prefactor_num = self.target.xi_dt * self.target.L_disk
        prefactor_denum = (
            np.power(4 * np.pi, 2) * self.target.epsilon_dt * m_e * np.power(c, 3)
        )

        integral_phi = np.trapz(_integrand, self.phi, axis=0)
        integral = np.trapz(integral_phi, self.l, axis=0)

        tau = prefactor_num / prefactor_denum * integral
        return tau.to_value("")

    def tau(self, nu):
        """optical depth

        .. math::
            \\tau_{\\gamma \\gamma}(\\nu)

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the opacity, **note** these are
            observed frequencies (observer frame).
        """
        if isinstance(self.target, SSDisk):
            return self._opacity_disk(nu)
        if isinstance(self.target, SphericalShellBLR):
            return self._opacity_shell_blr(nu)
        if isinstance(self.target, RingDustTorus):
            return self._opacity_ring_torus(nu)
