from __future__ import annotations

import math
import numpy as np
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "multivarious" / "multivarious" 


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

#preload dependency used by lognormal.py
_load_module(
    "multivarious.utl.correlated_rvs",
    PKG_DIR / "utl" / "correlated_rvs.py",
)

#lognormal.py itself
lognormal = _load_module(
    "multivarious.rvs.lognormal",
    PKG_DIR / "rvs" / "lognormal.py",
)



def median_from_mean_cov(mean: float, cov: float) -> float:
    if mean <= 0:
        raise ValueError("mean must be > 0 for a lognormal distribution.")
    if cov < 0:
        raise ValueError("cov must be >= 0.")
    sigma2 = math.log(1.0 + cov**2)
    mu = math.log(mean) - 0.5 * sigma2
    return math.exp(mu)


def simulate_p_v_gt_6(n_sims: int = 500_000, seed: int = 7) -> tuple[float, float]:
    mean_R, cov_R = 1.0, 1.0
    mean_N, cov_N = 1.0, 0.50

    med_R = median_from_mean_cov(mean_R, cov_R)
    med_N = median_from_mean_cov(mean_N, cov_N)

    rng = np.random.default_rng(seed)
    seed_R = int(rng.integers(0, 2**31 - 1))
    seed_N = int(rng.integers(0, 2**31 - 1))

    R = lognormal.rnd(med_R, cov_R, n_sims, seed=seed_R)
    N = lognormal.rnd(med_N, cov_N, n_sims, seed=seed_N)

    V = N * R * np.log(R + 1.0)

    hits = V > 6.0
    p_hat = float(np.mean(hits))
    mc_se = math.sqrt(p_hat * (1.0 - p_hat) / n_sims)

    return p_hat, mc_se


def main() -> None:
    n_sims = 500_000
    p_hat, mc_se = simulate_p_v_gt_6(n_sims=n_sims, seed=7)

    ci_lo = p_hat - 1.96 * mc_se
    ci_hi = p_hat + 1.96 * mc_se

    print("Problem 4.37(b) Monte Carlo")
    print(f"  sims = {n_sims:,}")
    print(f"  P(V > 6) ≈ {p_hat:.6f}")
    print(f"  MC SE   ≈ {mc_se:.6f}")
    print(f"  95% CI  ≈ [{ci_lo:.6f}, {ci_hi:.6f}]")


if __name__ == "__main__":
    main()
