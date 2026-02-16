from __future__ import annotations
import math
import numpy as np
import os
import sys
sys.path.insert(0, r"C:\Users\quint\Libraries\multivarious")

from multivarious.rvs import lognormal


def median_from_mean_cov(mean: float, cov: float) -> float:
    """
    Convert (mean, cov) of a lognormal variable into the *median* parameter.

    For a lognormal X:
        sigma_ln^2 = ln(1 + cov^2)
        mu_ln = ln(mean) - 0.5*sigma_ln^2
        median = exp(mu_ln)
    """
    if mean <= 0:
        raise ValueError("mean must be > 0 for a lognormal distribution.")
    if cov < 0:
        raise ValueError("cov must be >= 0.")
    sigma2 = math.log(1.0 + cov**2)
    mu = math.log(mean) - 0.5 * sigma2
    return math.exp(mu)


def simulate_p_v_gt_6(
    n_sims: int = 500_000,
    seed: int = 7,
) -> tuple[float, float]:
    """
    Monte Carlo estimate of P(V > 6), where
        V = N * R * ln(R + 1)

    Given in the problem:
        R is lognormal with mean 1 in and cov 1.00
        N is independent of R, mean 1 and cov 0.50
        (I model N as lognormal too since it's a multiplicative model-error factor.)
    Returns:
        (p_hat, mc_se)
    """
    mean_R, cov_R = 1.0, 1.0
    mean_N, cov_N = 1.0, 0.50

    # Convert to the (median, cov) parameterization used by multivarious.rvs.lognormal
    med_R = median_from_mean_cov(mean_R, cov_R)
    med_N = median_from_mean_cov(mean_N, cov_N)

    rng = np.random.default_rng(seed)

    # Draw samples (split the RNG stream so it's reproducible but still independent-looking)
    R = lognormal.rvs(med_R, cov_R, n=n_sims, seed=rng.integers(0, 2**31 - 1))
    N = lognormal.rvs(med_N, cov_N, n=n_sims, seed=rng.integers(0, 2**31 - 1))

    V = N * R * np.log(R + 1.0)

    hits = (V > 6.0)
    p_hat = float(np.mean(hits))

    # Monte Carlo standard error for a Bernoulli estimator
    mc_se = math.sqrt(p_hat * (1.0 - p_hat) / n_sims)

    return p_hat, mc_se


def main() -> None:
    n_sims = 500_000
    p_hat, mc_se = simulate_p_v_gt_6(n_sims=n_sims, seed=7)

    # 95% normal-approx CI just to sanity check precision
    ci_lo = p_hat - 1.96 * mc_se
    ci_hi = p_hat + 1.96 * mc_se

    print("Problem 4.37(b) Monte Carlo")
    print(f"  sims = {n_sims:,}")
    print(f"  P(V > 6) ≈ {p_hat:.6f}")
    print(f"  MC SE   ≈ {mc_se:.6f}")
    print(f"  95% CI  ≈ [{ci_lo:.6f}, {ci_hi:.6f}]")


if __name__ == "__main__":
    main()
