.. _synchrotron_polarization:


Synchrotron Polarization
=====================

Synchrotron radiation is linearly polarized. The computation of the polarization degree of the synchrotron radiation emitted by the relativistic particles (both electrons and protons) populating a blob can be performed using `agnpy`.  
The implementation relies on [Rybicki&Lightman1989](https://ui.adsabs.harvard.edu/abs/1979rpa..book.....R/abstract) and [DermerMenon2009](https://ui.adsabs.harvard.edu/abs/2009herb.book.....D/abstract).

It is here illustrated how to produce the wavelength-dependent polarization degree of the synchrotron radiation starting from a :class:`~agnpy.emission_regions.Blob`.

.. plot:: snippets/synchro_snippet.py
   :include-source:

For additional cases of study, and for validation of the ``agnpy`` implementation against ``jetset``, check the `tutorial notebook on synchrotron polarization <tutorials/synchrotron_polarization.html>`_.

.. toctree::
   :maxdepth: 1

   tutorials/synchrotron_polarization.ipynb
