from __future__ import annotations

import math
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np


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


def _ensure_pkg(name: str) -> None:
    """Create a package-like module in sys.modules so dotted imports work."""
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = []  # mark as package
    sys.modules[name] = m


def main() -> None:
    _ensure_pkg("multivarious")
    _ensure_pkg("multivarious.utl")
    _ensure_pkg("multivarious.fit")

    format_plot_mod = _load_module(
        "multivarious.utl.format_plot",
        PKG_DIR / "utl" / "format_plot.py",
    )
    if not hasattr(format_plot_mod, "format_plot"):
        raise AttributeError(
            "Expected function 'format_plot' inside format_plot.py, but it was not found."
        )
 
    sys.modules["multivarious.utl"].format_plot = format_plot_mod.format_plot  

   
    plot_ECDF_ci_mod = _load_module(
        "multivarious.utl.plot_ECDF_ci",
        PKG_DIR / "utl" / "plot_ECDF_ci.py",
    )
    sys.modules["multivarious.utl"].plot_ECDF_ci = plot_ECDF_ci_mod  

    
    poly_fit_mod = _load_module(
        "multivarious.fit.poly_fit",
        PKG_DIR / "fit" / "poly_fit.py",
    )
    poly_fit = poly_fit_mod.poly_fit

    X = np.array([5, 35, 20, 15, 4, 6, 18, 23, 38, 8, 12, 17, 17, 13, 7, 23], dtype=float)
    Y = np.array([1.5, 12, 7.5, 6.3, 1.2, 1.7, 7.2, 8, 14, 3.6, 3.7, 6.6, 4.4, 4.5, 2.8, 8], dtype=float)

    p = np.array([1.0])  # y = c * x

    # fig_no=1 -> saves poly_fit-0001.pdf, poly_fit-0002.pdf, poly_fit-0003.pdf
    c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, BIC, condNo = poly_fit(
        x=X,
        y=Y,
        p=p,
        fig_no=1,
        Sy=None,
        rof=None,
        b=0.0,
    )

    c_hat = float(c[0])
    se_c = float(Sc[0])

    r_xy = float(np.corrcoef(X, Y)[0, 1])

    sigma_r = math.sqrt(float(Vr))


    n = len(X)
    df = n - len(p) - 1  # Nd - Np - 1
    try:
        from scipy.stats import t as scipy_t
        tcrit = float(scipy_t.ppf(0.95, df=df))
    except Exception:
        tcrit = 1.761  # approx for df=14

    c_lo_90 = c_hat - tcrit * se_c
    c_hi_90 = c_hat + tcrit * se_c

    print("Problem 8.12 — Fit y_hat(x;c)=c x using multivarious.fit.poly_fit")
    print(f"  n = {n}")
    print(f"  c_hat = {c_hat:.6f}")
    print(f"  SE(c) = {se_c:.6f}")
    print(f"  90% CI for c (t, df={df}) = [{c_lo_90:.6f}, {c_hi_90:.6f}]")
    print(f"  Corr r(X,Y) = {r_xy:.6f}")
    print(f"  Residual SD sigma_r = {sigma_r:.6f}")
    print(f"  R^2 = {float(R2):.6f}")
    print(f"  condNo = {float(condNo):.3f}")
    print("  Saved figures: poly_fit-0001.pdf, poly_fit-0002.pdf, poly_fit-0003.pdf")


if __name__ == "__main__":
    main()
