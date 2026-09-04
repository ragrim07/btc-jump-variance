"""
The damping trio  f, g, ghat  and their derivatives (Appendix A of
affine_intensity_theory).

Origin. A stationary latent factor X(t) with autocovariance
Cov(X(t),X(s)) = s2 * exp(-a|t-s|) is never observed pointwise; we observe its
DAILY INTEGRAL  I_t = Int_{t-1}^{t} X(s) ds.  Integrating a mean-reverting
process over a unit window "blurs" its second-moment structure by three
universal factors that depend only on the mean-reversion rate a:

    f(a)  = 2 (a - 1 + e^-a) / a^2          variance blur   (k = 0)
    ghat(a) = (1 - e^-a) / a                one-sided blur  (half of a boundary)
    g(a)  = ghat(a)^2 = (1 - e^-a)^2 / a^2  lag blur        (k >= 1)

f multiplies the factor variance; g multiplies each lag-k autocovariance; the
jump they undergo between k=0 and k=1 (f -> g) is the structural
variance-to-autocovariance discontinuity mandated by the model. ghat is the
one-sided integral that appears in the co-jump (triangle) geometry.

All expressions use expm1/log1p so they stay accurate as a -> 0.
"""

import numpy as np


def f(a):
    """Variance damping f(a) = 2(a - 1 + e^-a)/a^2.  Limits: a->0: 1 - a/3; a->inf: 2/a."""
    return 2.0 * (a + np.expm1(-a)) / a ** 2


def f_prime(a):
    """f'(a) = 2(1 - e^-a)/a^2 - 4(a - 1 + e^-a)/a^3."""
    return 2.0 * (-np.expm1(-a)) / a ** 2 - 4.0 * (a + np.expm1(-a)) / a ** 3


def ghat(a):
    """One-sided damping ghat(a) = (1 - e^-a)/a.  Limits: a->0: 1 - a/2; a->inf: 1/a."""
    return -np.expm1(-a) / a


def ghat_prime(a):
    """ghat'(a) = e^-a/a - (1 - e^-a)/a^2."""
    return np.exp(-a) / a + np.expm1(-a) / a ** 2


def g(a):
    """Lag damping g(a) = ghat(a)^2 = (1 - e^-a)^2/a^2.  Limits: a->0: 1 - a; a->inf: 1/a^2."""
    return ghat(a) ** 2


def g_prime(a):
    """g'(a) = 2 ghat(a) ghat'(a)  (chain rule on g = ghat^2)."""
    return 2.0 * ghat(a) * ghat_prime(a)


def lag_term(a, k):
    """m_k(a) = g(a) * e^{-a(k-1)}  -- the full lag-k autocovariance shape."""
    return g(a) * np.exp(-a * (k - 1.0))


def lag_term_prime(a, k):
    """d/da [ g(a) e^{-a(k-1)} ] = [g'(a) - (k-1) g(a)] e^{-a(k-1)}."""
    return (g_prime(a) - (k - 1.0) * g(a)) * np.exp(-a * (k - 1.0))
