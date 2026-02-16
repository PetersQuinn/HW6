from __future__ import annotations

import math
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "multivarious" / "multivarious"
DATA_PATH = HERE / "cee251-hw6-2026.txt"


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_pkg(name: str) -> None:
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = []  # mark as package
    sys.modules[name] = m


def _pdf_to_png(pdf_path: Path, dpi: int = 300) -> Path:
    """
    Convert a 1-page PDF to a PNG next to it using PyMuPDF (fitz).
    Returns the output PNG path.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF is required to convert PDF->PNG. Install with:\n"
            "  pip install pymupdf\n"
            f"Original import error: {e}"
        )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if doc.page_count < 1:
        raise ValueError(f"PDF has no pages: {pdf_path}")

    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    out_path = pdf_path.with_suffix(".png")
    pix.save(str(out_path))
    doc.close()
    return out_path


EMBEDDED_DATA = """
1.1068170746835337 3.2054629307412714
1.1051349184810992 3.2037684333238134
1.8166230563010115 4.1209675955935934
1.5347523283250686 3.7160318368012017
1.5333671682851624 3.7141576808581367
2.2891702146532067 4.8823920077650822
1.5076821429322533 3.6796321146256266
2.3024813653500487 4.9049938506072657
2.1170558548530272 4.5952683290323915
1.9156716267319842 4.2730719747546013
2.4075716403836069 5.0852442710820549
2.7477679290485995 5.6877479943853304
2.5392798576028905 5.3153293610578984
1.3337789580531745 3.4579970127576258
2.9443189840724866 6.046551652717655
1.5897359636830988 3.7914103151981124
1.8251936857937463 4.1339466363460531
2.4391031158191439 5.1399234107907237
1.7989424827810740 4.0943063268775708
1.7031298147313623 3.9525975863489373
1.6371555589184892 3.8579088955409224
1.6483389266753912 3.873785401439898
2.0666790897104335 4.5131747692806892
0.9908472502155852 3.0954805181845897
2.0550458296272223 4.4943525550498817
1.8572916861844331 4.1828678088063924
2.5157674097199587 5.2739374625197222
1.6075442209691717 3.8162257333169221
2.8003663475114657 5.7830960622937697
1.4748363140461218 3.6361211620154368
2.0095604097757724 4.4212635827361462
2.3815609836155902 5.0403406280537473
1.9970561710723751 4.4013151645968147
1.2959127329128928 3.4127694141981859
1.6642155712002749 3.8964485108160618
2.4446983689940307 5.1496536324040676
2.4773699470453323 5.2066311878834766
1.0924012432768222 3.1910328604834475
2.2332560128165952 4.7880486823094328
1.8980051322111966 4.2456121243583027
2.5899392286126681 5.4049522551126579
2.3309191440090822 4.9534573661367789
1.8076772778348034 4.1074586607571302
1.5139422222884473 3.688006892759522
3.6960201577691594 7.4665066703328415
1.9716191575236262 4.3609320239576608
1.9033615225190579 4.2539231001735409
1.7248204218458922 3.9842559955929135
2.4728337326511372 5.1987039875448477
4.4834255502919111 9.0007350697761321
3.3083965910106365 6.7265174952618141
1.4472981384305552 3.6002110413619088
1.8140949746728094 4.1171459919342119
3.4871462259431039 7.0660565691475572
1.6488221343181584 3.8744730213109593
2.5681046860860466 5.3662515977551362
1.9100532029734061 4.2643239314113028
1.9942460572921343 4.3968408325934369
2.6252113118921918 5.4676959775923573
1.4037186621453852 3.5444822924761272
2.4321622218278787 5.1278644928600663
1.9944255559637452 4.3971265384622793
3.0969093005873081 6.3293845551444985
2.0323437843049308 4.4577717470139362
1.8389846596632120 4.1549056045381985
1.9347080949518369 4.3028154893548116
2.0864348261857866 4.5452563192235846
2.6748435940480673 5.5564400081230065
2.4230625194319035 5.1120741684188644
3.9310107345968248 7.9208929533258372
1.9230456280235728 4.2845746494082428
1.7506427087552674 4.0222722974065164
1.3421934043781711 3.4682027024271136
1.4531346407512753 3.6077777187438431
2.3618443032509888 5.0064273389701945
2.2249546523067227 4.7741262626757459
1.7665778543922923 4.0459063837206557
1.1411409659385585 3.2406449137695681
3.1328521248336498 6.3964841085143593
2.6618854039048245 5.5332202038305258
1.6759824116139959 3.9133375915581672
1.9745106751902606 4.3655090350129111
1.7872498592806334 4.076759615331575
2.9610356653716474 6.077367164430032
2.1241255220301016 4.6068634103968771
1.9250655872501403 4.2877297432409467
1.6939367914803187 3.9392571942619341
2.0840298397244261 4.541343008991662
2.4418460610658643 5.1446924090391875
1.8923645872918240 4.2368741996823065
1.2588502183087231 3.3696414994364341
2.1768004584787519 4.6938122897647458
1.5261940536307026 3.7044721894471406
2.6195444102106245 5.4575969362964907
3.0181390437756916 6.1829544470974458
2.8362349230089690 5.8484080192101606
2.7545860374335214 5.7000781454852545
1.6553608185356059 3.8837909928334176
3.0287208379755359 6.2025741195533826
1.4450703278092395 3.5973291485194165
""".strip()


def load_xy() -> tuple[np.ndarray, np.ndarray]:
    if DATA_PATH.exists():
        raw = DATA_PATH.read_text(encoding="utf-8").strip().splitlines()
        pairs: list[tuple[float, float]] = []
        for line in raw:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("%"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            pairs.append((float(parts[0]), float(parts[1])))
        if len(pairs) < 5:
            raise ValueError(f"Found too few valid rows in {DATA_PATH}")
        x = np.array([a for a, _ in pairs], dtype=float)
        y = np.array([b for _, b in pairs], dtype=float)
        return x, y

    pairs = []
    for line in EMBEDDED_DATA.splitlines():
        a, b = line.split()
        pairs.append((float(a), float(b)))
    x = np.array([a for a, _ in pairs], dtype=float)
    y = np.array([b for _, b in pairs], dtype=float)
    return x, y


def percent_param_errors(c: np.ndarray, Sc: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(c) < 1e-12, np.nan, np.abs(c))
    return 100.0 * Sc / denom


def main() -> None:
    _ensure_pkg("multivarious")
    _ensure_pkg("multivarious.utl")
    _ensure_pkg("multivarious.fit")

    format_plot_mod = _load_module(
        "multivarious.utl.format_plot",
        PKG_DIR / "utl" / "format_plot.py",
    )
    if not hasattr(format_plot_mod, "format_plot"):
        raise AttributeError("format_plot.py does not define format_plot(...)")
    sys.modules["multivarious.utl"].format_plot = format_plot_mod.format_plot  

    _load_module(
        "multivarious.utl.plot_ECDF_ci",
        PKG_DIR / "utl" / "plot_ECDF_ci.py",
    )
    poly_fit_mod = _load_module(
        "multivarious.fit.poly_fit",
        PKG_DIR / "fit" / "poly_fit.py",
    )
    poly_fit = poly_fit_mod.poly_fit

    x, y = load_xy()
    n = len(x)

    candidate_p_sets: list[list[float]] = [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 0.5, 1],
        [0, 0.5, 1, 2],
        [0, 0.5, 1, 1.5, 2],
        [0, 1, 1.5, 2],
        [0.5, 1, 1.5],
        [0.5, 1, 2],
        [1, 2],
        [1, 1.5, 2],
        [-1, 0, 1],
        [-0.5, 0, 0.5, 1],
    ]

    best = None  

    print("\nSearching for exponent set p with all parameter % errors < 20% ...\n")
    chosen_p = None
    for p_list in candidate_p_sets:
        p = np.array(p_list, dtype=float)
        try:
            c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, BIC, condNo = poly_fit(
                x=x,
                y=y,
                p=p,
                fig_no=0,
                Sy=None,
                rof=None,
                b=0.0,
            )
        except Exception as e:
            print(f"  p={p_list} -> failed: {e}")
            continue

        pct = percent_param_errors(c, Sc)
        max_pct = float(np.nanmax(pct))
        print(f"  p={p_list} -> max %SE = {max_pct:6.2f}%   (R^2={float(R2):.4f})")

        if best is None or max_pct < best[0]:
            best = (max_pct, p_list, (c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, BIC, condNo))

        if max_pct < 20.0:
            chosen_p = p
            break

    if chosen_p is None:
        if best is None:
            raise RuntimeError("No successful fits. Check data and multivarious paths.")
        _, best_p_list, _ = best
        chosen_p = np.array(best_p_list, dtype=float)
        print("\nNo candidate achieved <20% for all parameters; using best candidate found.")

    print("\nRunning final fit with plots enabled...\n")
    c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, BIC, condNo = poly_fit(
        x=x,
        y=y,
        p=chosen_p,
        fig_no=1,
        Sy=None,
        rof=None,
        b=0.0,
    )

    pct = percent_param_errors(c, Sc)
    sigma_r = math.sqrt(float(Vr))

    print("\n" + "=" * 70)
    print("Final Model Summary (for HW write-up)")
    print("=" * 70)
    print(f"Data points: n = {n}")
    print(f"Chosen exponents p = {list(map(float, chosen_p))}")
    print(f"Residual SD (sigma_r) = {sigma_r:.6f}")
    print(f"R^2 = {float(R2):.6f}")
    print(f"condNo = {float(condNo):.3f}")
    print("\nParameter table:")
    print("   i      p_i         c_i            SE(c_i)        %SE")
    for i, (pi, ci, sci, pei) in enumerate(zip(chosen_p, c, Sc, pct), start=1):
        print(f"  {i:2d}   {pi:8.3f}   {ci:12.6e}   {sci:12.6e}   {pei:7.2f}%")


    pdfs = [
        HERE / "poly_fit-0001.pdf",
        HERE / "poly_fit-0002.pdf",
        HERE / "plot_ECDF-0003.pdf",  
    ]

    print("\nConverting PDFs to PNGs (dpi=300)...")
    for pdf in pdfs:
        try:
            out_png = _pdf_to_png(pdf, dpi=300)
            print(f"  Saved: {out_png.name}")
        except Exception as e:
            print(f"  Could not convert {pdf.name} -> PNG: {e}")

    print("\nSaved figures from poly_fit:")
    for pdf in pdfs:
        print(f"  {pdf.name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
