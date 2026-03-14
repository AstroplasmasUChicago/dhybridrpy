"""Plasma dispersion relations for dHybridR analysis.

Conventions (all quantities normalised to ion scales):
  ω  → normalised to Ω_ci   (ion cyclotron frequency)
  k  → normalised to Ω_ci / v_A   (Alfvén unit)
  v  → normalised to v_A
  c  → normalised to v_A
  plus_minus = ±1  (controls right-/left-hand polarisation branch)
"""

import numpy as np
from scipy.special import wofz

# Small imaginary offset added to avoid poles exactly on the real axis when
# evaluating the CR susceptibility at ω′ = ∓Ω/γ_iso.
_EPS_IMAG = 1j * 1e-10


def Z_pade(zeta: complex) -> complex:
    """J-pole Padé approximation of the plasma dispersion function Z(ζ).

    The plasma dispersion function is defined as

        Z(ζ) = (1/√π) ∫_{-∞}^{∞} exp(-t²) / (t - ζ) dt   Im(ζ) > 0

    and analytically continued below the real axis.  This implementation
    uses the exact relation

        Z(ζ) = i√π · w(ζ)

    where w(ζ) = exp(-ζ²) erfc(-iζ) is the Faddeeva function evaluated
    via ``scipy.special.wofz``.  The result is smooth for all complex ζ
    and is therefore suitable for use inside a root-finder.
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def D_cold_plasma(w, k, params) -> complex:
    """Cold plasma dispersion relation (non-relativistic electrons, ions, and CRs)."""

    plus_minus = params['plus_minus']
    c = params['c']
    n_e = params['n_e']
    n_i = params['n_i']
    n_cr = params['n_cr']
    vdrift_e = params['vdrift_e']
    vdrift_cr = params['vdrift_cr']
    me_over_mi = params['me_over_mi']

    lhs = (w**2) / (c**2) - k**2
    term1 = (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (w - k*vdrift_e - plus_minus * (1 / me_over_mi))
    term2 = w / (w + plus_minus)
    term3 = (n_cr / n_i) * (w - k*vdrift_cr) / (w - k*vdrift_cr + plus_minus)

    return lhs - (term1 + term2 + term3)


def D_cold_plasma_relativistic_beam_crs(w, k, params) -> complex:
    """Cold plasma dispersion relation with relativistic CR beam (my derivation)."""

    plus_minus = params['plus_minus']
    c = params['c']
    n_e = params['n_e']
    n_i = params['n_i']
    n_cr = params['n_cr']
    vdrift_e = params['vdrift_e']
    vdrift_cr = params['vdrift_cr']
    gammadrift_cr = params['gammadrift_cr']
    me_over_mi = params['me_over_mi']

    lhs = (w**2) / (c**2) - k**2
    term1 = (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (w - k*vdrift_e - plus_minus * (1 / me_over_mi))
    term2 = w / (w + plus_minus)
    term3 = (n_cr / n_i / gammadrift_cr) * (w - k * vdrift_cr) / (w - k * vdrift_cr + plus_minus/gammadrift_cr)

    return lhs - (term1 + term2 + term3)


def D_warm_plasma_relativistic_beam_crs(w, k, params) -> complex:
    """Dispersion relation with relativistic CR beam and warm ions + electrons.

    Uses Z_pade (J-pole Padé) for the ion susceptibility — rational
    function that is smooth everywhere and root-finder friendly.
    """

    plus_minus = params['plus_minus']
    c = params['c']
    n_e = params['n_e']
    n_i = params['n_i']
    n_cr = params['n_cr']
    vdrift_e = params['vdrift_e']
    vdrift_cr = params['vdrift_cr']
    gammadrift_cr = params['gammadrift_cr']
    vth_i = params['vth_i']
    vth_e = params['vth_e']
    me_over_mi = params['me_over_mi']

    zeta_e = (w - k * vdrift_e + plus_minus * (-1.0 / me_over_mi)) / (np.sqrt(2) * k * vth_e)
    zeta_i = (w + plus_minus) / (np.sqrt(2) * k * vth_i)

    lhs = (w**2) / (c**2) - k**2
    term1 = -(1.0 / np.sqrt(2)) * (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (k * vth_e) * Z_pade(zeta_e)
    term2 = -(1.0 / np.sqrt(2)) * w / (k * vth_i) * Z_pade(zeta_i)
    term3 = (n_cr / n_i / gammadrift_cr) * (w - k * vdrift_cr) / (w - k * vdrift_cr + plus_minus/gammadrift_cr)

    return lhs - (term1 + term2 + term3)


def D_warm_plasma_relativistic_monoenergetic_crs(w, k, params) -> complex:
    """Dispersion relation with relativistic CR isotropic + monoenergetic distribution and warm ions + electrons.

    Uses Z_pade (J-pole Padé) for the ion susceptibility — rational
    function that is smooth everywhere and root-finder friendly.
    """

    plus_minus = params['plus_minus']
    c = params['c']
    n_e = params['n_e']
    n_i = params['n_i']
    n_cr = params['n_cr']
    vdrift_e = params['vdrift_e']
    vdrift_cr = params['vdrift_cr']
    gammadrift_cr = params['gammadrift_cr']
    vth_i = params['vth_i']
    vth_e = params['vth_e']
    me_over_mi = params['me_over_mi']
    viso_cr = params['viso_cr']
    gammaiso_cr = params['gammaiso_cr']

    w_prime = gammadrift_cr * (w - vdrift_cr * k)
    k_prime = gammadrift_cr * (k - vdrift_cr * w / c**2)

    a = w_prime + _EPS_IMAG + plus_minus / gammaiso_cr
    b = k_prime * viso_cr

    zeta_e = (w - k * vdrift_e + plus_minus * (-1.0 / me_over_mi)) / (np.sqrt(2) * k * vth_e)
    zeta_i = (w + plus_minus) / (np.sqrt(2) * k * vth_i)

    lhs = (w**2) / (c**2) - k**2
    term1 = -(1.0 / np.sqrt(2)) * (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (k * vth_e) * Z_pade(zeta_e)
    term2 = -(1.0 / np.sqrt(2)) * w / (k * vth_i) * Z_pade(zeta_i)

    if np.isclose(viso_cr, 0.0):
        # Cold-beam limit of the isotropic formula below
        term3 = (n_cr / n_i / gammadrift_cr) * w_prime / a
    else:
        term3 = -(n_cr / n_i / gammadrift_cr / 4.0) * w_prime * ((1 + 2*gammaiso_cr**2) / b**3 / gammaiso_cr**3) * ( (a**2 - b**2) * np.log((a+b)/(a-b)) - 2*a*b )

    return lhs - (term1 + term2 + term3)


def D_plasma_relativistic_crs(w, k, params) -> complex:
    """General dispersion relation with relativistic CRs.

    Selects warm or cold susceptibilities for ions and electrons independently,
    and selects the appropriate cosmic-ray term based on whether the CR
    distribution is a beam or a monoenergetic isotropic shell.

    Electron susceptibility
    -----------------------
    Warm  (``params['warm_electrons'] = True``, default):
        term1 = -(1/√2) * (n_e/n_i) * (1/me_over_mi) * (ω - k·v_de)/(k·v_th_e) * Z(ζ_e)
    Cold  (``params['warm_electrons'] = False``):
        term1 = (n_e/n_i) * (1/me_over_mi) * (ω - k·v_de) / (ω - k·v_de - ±(1/me_over_mi))

    Ion susceptibility
    ------------------
    Warm  (``params['warm_ions'] = True``, default):
        term2 = -(1/√2) * ω/(k·v_th_i) * Z(ζ_i)
    Cold  (``params['warm_ions'] = False``):
        term2 = ω / (ω ± 1)

    CR susceptibility
    -----------------
    Beam  (``viso_cr ≈ 0``):
        term3 = (n_cr / n_i / γ_d) * ω' / a         a = ω' + i·0⁺ + ±/γ_iso
    Monoenergetic isotropic  (``viso_cr > 0``):
        term3 = -(n_cr / n_i / γ_d / 4) * ω' * [(1+2γ_iso²)/(b³γ_iso³)] * [(a²-b²)·ln((a+b)/(a-b)) - 2ab]

    Required params keys
    --------------------
    Always:  plus_minus, c, n_e, n_i, n_cr, vdrift_e, vdrift_cr, gammadrift_cr,
             me_over_mi, viso_cr, gammaiso_cr
    If warm_electrons=True (default):  vth_e
    If warm_ions=True (default):       vth_i
    """

    plus_minus = params['plus_minus']
    c = params['c']
    n_e = params['n_e']
    n_i = params['n_i']
    n_cr = params['n_cr']
    vdrift_e = params['vdrift_e']
    vdrift_cr = params['vdrift_cr']
    gammadrift_cr = params['gammadrift_cr']
    me_over_mi = params['me_over_mi']
    viso_cr = params['viso_cr']
    gammaiso_cr = params['gammaiso_cr']
    warm_ions = params.get('warm_ions', True)
    warm_electrons = params.get('warm_electrons', True)

    w_prime = gammadrift_cr * (w - vdrift_cr * k)
    k_prime = gammadrift_cr * (k - vdrift_cr * w / c**2)

    a = w_prime + _EPS_IMAG + plus_minus / gammaiso_cr
    b = k_prime * viso_cr

    lhs = (w**2) / (c**2) - k**2

    # --- electron susceptibility ---
    if warm_electrons:
        vth_e = params['vth_e']
        zeta_e = (w - k * vdrift_e + plus_minus * (-1.0 / me_over_mi)) / (np.sqrt(2) * k * vth_e)
        term1 = -(1.0 / np.sqrt(2)) * (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (k * vth_e) * Z_pade(zeta_e)
    else:
        term1 = (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (w - k*vdrift_e - plus_minus * (1 / me_over_mi))

    # --- ion susceptibility ---
    if warm_ions:
        vth_i = params['vth_i']
        zeta_i = (w + plus_minus) / (np.sqrt(2) * k * vth_i)
        term2 = -(1.0 / np.sqrt(2)) * w / (k * vth_i) * Z_pade(zeta_i)
    else:
        term2 = w / (w + plus_minus)

    # --- CR susceptibility ---
    if np.isclose(viso_cr, 0.0):
        # Cold-beam limit of the isotropic formula below
        term3 = (n_cr / n_i / gammadrift_cr) * w_prime / a
    else:
        term3 = -(n_cr / n_i / gammadrift_cr / 4.0) * w_prime * ((1 + 2*gammaiso_cr**2) / b**3 / gammaiso_cr**3) * ( (a**2 - b**2) * np.log((a+b)/(a-b)) - 2*a*b )

    return lhs - (term1 + term2 + term3)


# def D_warm_plasma_relativistic_crs(w, k, params) -> complex:
#     """Dispersion relation with relativistic CR beam and warm ions.
#
#     Uses Z_pade (J-pole Padé) for the ion susceptibility — rational
#     function that is smooth everywhere and root-finder friendly.
#     """
#
#     plus_minus = params['plus_minus']
#     c = params['c']
#     n_e = params['n_e']
#     n_i = params['n_i']
#     n_cr = params['n_cr']
#     vdrift_e = params['vdrift_e']
#     vdrift_cr = params['vdrift_cr']
#     gammadrift_cr = params['gammadrift_cr']
#     vth_i = params['vth_i']
#     vth_e = params['vth_e']
#     me_over_mi = params['me_over_mi']
#
#     zeta_e = (w - k * vdrift_e + plus_minus * (-1.0 / me_over_mi)) / (np.sqrt(2) * k * vth_e)
#     zeta_i = (w + plus_minus) / (np.sqrt(2) * k * vth_i)
#
#     lhs = (w**2) / (c**2) - k**2
#     # term1 = (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (w - k*vdrift_e - plus_minus * (1 / me_over_mi))
#     term1 = -(1.0 / np.sqrt(2)) * (n_e / n_i) * (1 / me_over_mi) * (w - k*vdrift_e) / (k * vth_e) * Z_pade(zeta_e)
#     term2 = -(1.0 / np.sqrt(2)) * w / (k * vth_i) * Z_pade(zeta_i)
#     term3 = (n_cr / n_i / gammadrift_cr) * (w - k * vdrift_cr) / (w - k * vdrift_cr + plus_minus/gammadrift_cr)
#
#     return lhs - (term1 + term2 + term3)
