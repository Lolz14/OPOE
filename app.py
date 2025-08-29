import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opoe
import math
import time

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
# ---------------------------
# Extra math helpers
# ---------------------------
def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def vega_bs(S0, K, r, T, sigma):
    # Vectorized Black–Scholes Vega (per 1.0 change in sigma, not 1%)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * _norm_pdf(d1) * np.sqrt(T)

def vega_grid(S0, K_grid, T_grid, r, sigma_grid):
    # sigma_grid shape [nT, nK]; K_grid [nK], T_grid [nT]
    Tm = T_grid[:, None]
    Km = K_grid[None, :]
    return vega_bs(S0, Km, r, Tm, sigma_grid)

# ---------------------------
# Build price and IV surfaces correctly (rebuild pricer per (K,T))
# ---------------------------
def compute_surfaces(model, rate, S0=100.0,
                     strikes=np.linspace(80, 120, 9),
                     maturities=np.linspace(0.25, 2.0, 8),
                     N=10, Kpoly=10,
                     fft_kwargs=None):
    """
    Returns dict with:
      prices_fft, prices_ope, iv_fft, iv_ope  (each shape [nT, nK]),
      strikes, maturities
    """
    if fft_kwargs is None:
        fft_kwargs = dict(Npow=11, A=1200)  # slightly larger grid/range than before

    nT, nK = len(maturities), len(strikes)
    prices_fft = np.full((nT, nK), np.nan)
    prices_ope = np.full((nT, nK), np.nan)
    iv_fft     = np.full((nT, nK), np.nan)
    iv_ope     = np.full((nT, nK), np.nan)

    OPEClass = get_ope_class(N)

    for iT, T in enumerate(maturities):
        for iK, K_strike in enumerate(strikes):
            payoff = opoe.EuropeanCallPayoff(float(K_strike))

            # Build pricers per (K,T)
            FFTpricer = opoe.FFTOptionPricer(T, rate, payoff, model, **fft_kwargs)
            OPEpricer = OPEClass(T, rate, payoff, model, K=Kpoly, solving_param=opoe.OPEMethod.Direct)

            try:
                p_fft = FFTpricer.price()
                prices_fft[iT, iK] = p_fft
                iv_fft[iT, iK] = implied_vol_call(p_fft, S0, K_strike, rate, T)
            except Exception:
                pass

            try:
                p_ope = OPEpricer.price()
                prices_ope[iT, iK] = p_ope
                iv_ope[iT, iK] = implied_vol_call(p_ope, S0, K_strike, rate, T)
            except Exception:
                pass

    return dict(prices_fft=prices_fft, prices_ope=prices_ope,
                iv_fft=iv_fft, iv_ope=iv_ope,
                
        
                strikes=np.array(strikes), maturities=np.array(maturities))


def plot_surface_comparison_fixed(surf, S0=100.0, N=10, Kpoly=10):
    K = surf["strikes"]; T = surf["maturities"]
    X, Y = np.meshgrid(K, T)
    Zfft, Zope = surf["iv_fft"], surf["iv_ope"]

    # Side-by-side IV surfaces
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    s1 = ax1.plot_surface(X, Y, Zfft, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_title("Implied Vol Surface (FFT)")
    ax1.set_xlabel("Strike"); ax1.set_ylabel("Maturity"); ax1.set_zlabel("IV")
    fig.colorbar(s1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(122, projection="3d")
    s2 = ax2.plot_surface(X, Y, Zope, cmap="plasma", edgecolor="none", alpha=0.9)
    ax2.set_title(f"Implied Vol Surface (OPE-Direct, N={N}, K={Kpoly})")
    ax2.set_xlabel("Strike"); ax2.set_ylabel("Maturity"); ax2.set_zlabel("IV")
    fig.colorbar(s2, ax=ax2, shrink=0.6)

    plt.tight_layout(); plt.savefig("vol_surface_fft_vs_ope_FIXED.png")

    # Difference surface
    diff = Zope - Zfft
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    s3 = ax.plot_surface(X, Y, diff, cmap="coolwarm", edgecolor="none", alpha=0.95)
    ax.set_title("Difference Surface (OPE − FFT)")
    ax.set_xlabel("Strike"); ax.set_ylabel("Maturity"); ax.set_zlabel("ΔIV")
    fig.colorbar(s3, shrink=0.6)
    plt.tight_layout(); plt.savefig("vol_surface_difference_FIXED.png")

def nearest_index(arr, x):
    return int(np.argmin(np.abs(arr - x)))

def plot_smile_slices(surf, T_list=(0.25, 1.0, 2.0)):
    K = surf["strikes"]; T = surf["maturities"]
    for T0 in T_list:
        j = nearest_index(T, T0)
        plt.figure(figsize=(7,4))
        plt.plot(K, surf["iv_fft"][j,:], "o-", label="FFT")
        plt.plot(K, surf["iv_ope"][j,:], "s-", label="OPE")
        plt.title(f"Implied Vol Smile @ T={T[j]:.2f}")
        plt.xlabel("Strike"); plt.ylabel("IV"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"slice_smile_T{T[j]:.2f}.png")

        # ΔIV below
        plt.figure(figsize=(7,3.2))
        plt.plot(K, (surf["iv_ope"][j,:] - surf["iv_fft"][j,:]), "k-", lw=1.6)
        plt.axhline(0, color="gray", ls="--")
        plt.title(f"ΔIV (OPE−FFT) @ T={T[j]:.2f}")
        plt.xlabel("Strike"); plt.ylabel("ΔIV"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"slice_deltaIV_T{T[j]:.2f}.png")

def plot_term_slices(surf, K_list=(85, 100, 115)):
    K = surf["strikes"]; T = surf["maturities"]
    for K0 in K_list:
        i = nearest_index(K, K0)
        plt.figure(figsize=(7,4))
        plt.plot(T, surf["iv_fft"][:,i], "o-", label="FFT")
        plt.plot(T, surf["iv_ope"][:,i], "s-", label="OPE")
        plt.title(f"Term Structure @ K={K[i]:.0f}")
        plt.xlabel("Maturity"); plt.ylabel("IV"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(f"slice_term_K{K[i]:.0f}.png")

        plt.figure(figsize=(7,3.2))
        plt.plot(T, (surf["iv_ope"][:,i] - surf["iv_fft"][:,i]), "k-", lw=1.6)
        plt.axhline(0, color="gray", ls="--")
        plt.title(f"ΔIV (OPE−FFT) @ K={K[i]:.0f}")
        plt.xlabel("Maturity"); plt.ylabel("ΔIV"); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"slice_deltaIV_K{K[i]:.0f}.png")

def plot_total_variance_heatmaps(surf):
    K = surf["strikes"]; T = surf["maturities"]
    w_fft = (surf["iv_fft"]**2) * T[:,None]
    w_ope = (surf["iv_ope"]**2) * T[:,None]
    w_diff = w_ope - w_fft

    def heat(img, title, fname, cmap="viridis"):
        plt.figure(figsize=(6.5,4.6))
        plt.imshow(img, origin="lower", aspect="auto",
                   extent=[K.min(), K.max(), T.min(), T.max()], cmap=cmap)
        plt.colorbar(); plt.title(title); plt.xlabel("Strike"); plt.ylabel("Maturity")
        plt.tight_layout(); plt.savefig(fname)

    heat(w_fft, "Total Variance w=σ²T (FFT)", "heat_w_fft.png")
    heat(w_ope, "Total Variance w=σ²T (OPE)", "heat_w_ope.png")
    heat(w_diff, "Δw (OPE−FFT)", "heat_w_diff.png", cmap="coolwarm")

def plot_error_maps(surf, S0=100.0, r=0.05):
    K = surf["strikes"]; T = surf["maturities"]
    d_sigma = surf["iv_ope"] - surf["iv_fft"]
    d_price = surf["prices_ope"] - surf["prices_fft"]

    sigma_mid = 0.5 * (surf["iv_ope"] + surf["iv_fft"])
    V = vega_grid(S0, K, T, r, sigma_mid)
    price_equiv = d_sigma * V

    def heat(img, title, fname, cmap="coolwarm"):
        plt.figure(figsize=(6.5,4.6))
        plt.imshow(img, origin="lower", aspect="auto",
                   extent=[K.min(), K.max(), T.min(), T.max()], cmap=cmap)
        plt.colorbar(); plt.title(title); plt.xlabel("Strike"); plt.ylabel("Maturity")
        plt.tight_layout(); plt.savefig(fname)

    heat(d_sigma, "IV Difference Δσ (OPE−FFT)", "heat_dsigma.png")
    heat(d_price, "Price Difference ΔC (OPE−FFT)", "heat_dprice.png")
    heat(np.abs(d_price)/np.maximum(surf["prices_fft"], 1e-10),
         "Relative Price Error |ΔC|/C (vs FFT)", "heat_relprice.png")
    heat(price_equiv, "Price-Equivalent Error Δσ·Vega", "heat_price_equiv.png")

    # Scatter Δσ vs Vega
    plt.figure(figsize=(6.5,4.6))
    plt.scatter(V.flatten(), d_sigma.flatten(), s=12, alpha=0.6)
    plt.xlabel("Vega (BS, mid-σ)"); plt.ylabel("ΔIV (OPE−FFT)")
    plt.title("ΔIV vs Vega (low Vega ⇒ vol diffs blow up)")
    plt.grid(True); plt.tight_layout(); plt.savefig("scatter_dsigma_vs_vega.png")

def arbitrage_diagnostics(surf, r=0.05):
    K = surf["strikes"]; T = surf["maturities"]
    C_fft = surf["prices_fft"]; C_ope = surf["prices_ope"]

    # Assume uniform strike grid (linspace as in your code)
    dK = float(K[1] - K[0])

    def second_diff_K(C):
        # ∂²C/∂K² (discrete, uniform spacing)
        return (C[:, :-2] - 2.0 * C[:, 1:-1] + C[:, 2:]) / (dK**2)

    def calendar_diff(C):
        # C increases in T; adjacent differences
        return C[1:, :] - C[:-1, :]

    # Butterfly convexity (positive)
    bf_fft = second_diff_K(C_fft)
    bf_ope = second_diff_K(C_ope)

    # Risk-neutral density q(K,T) ≈ e^{rT} ∂²C/∂K²
    Tmid = T[:, None][:, 1:-1]
    q_fft = np.exp(r * Tmid) * bf_fft
    q_ope = np.exp(r * Tmid) * bf_ope

    # Calendar monotonicity
    cal_fft = calendar_diff(C_fft)
    cal_ope = calendar_diff(C_ope)

    def heat_generic(img, title, fname, Ktrim=1):
        # Adjust extents for interior arrays (due to second diff)
        plt.figure(figsize=(6.5,4.6))
        extent = [K[Ktrim], K[-Ktrim-1], T[0], T[-1]] if img.shape[1] == len(K)-2 else \
                 [K[0], K[-1], T[1], T[-1]]
        plt.imshow(img, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        plt.colorbar(); plt.title(title); plt.xlabel("Strike"); plt.ylabel("Maturity")
        plt.tight_layout(); plt.savefig(fname)

    heat_generic(bf_fft, "Butterfly Convexity ∂²C/∂K² (FFT)", "arb_butterfly_fft.png")
    heat_generic(bf_ope, "Butterfly Convexity ∂²C/∂K² (OPE)", "arb_butterfly_ope.png")
    heat_generic(q_fft, "Risk-Neutral Density q (FFT)", "arb_density_fft.png")
    heat_generic(q_ope, "Risk-Neutral Density q (OPE)", "arb_density_ope.png")

    # Violations: negative density / negative butterfly; negative calendar increments
    heat_generic(np.minimum(q_fft, 0.0), "Negative q regions (FFT)", "arb_negq_fft.png")
    heat_generic(np.minimum(q_ope, 0.0), "Negative q regions (OPE)", "arb_negq_ope.png")

    plt.figure(figsize=(6.5,4.6))
    extent = [K[0], K[-1], T[0], T[-2]]
    plt.imshow(np.minimum(cal_fft, 0.0), origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    plt.colorbar(); plt.title("Negative Calendar Increments (FFT)"); plt.xlabel("Strike"); plt.ylabel("T→")
    plt.tight_layout(); plt.savefig("arb_calendar_fft.png")

    plt.figure(figsize=(6.5,4.6))
    plt.imshow(np.minimum(cal_ope, 0.0), origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    plt.colorbar(); plt.title("Negative Calendar Increments (OPE)"); plt.xlabel("Strike"); plt.ylabel("T→")
    plt.tight_layout(); plt.savefig("arb_calendar_ope.png")
def sweep_ope_order(model, rate, S0=100.0,
                    strikes=np.linspace(80,120,9),
                    maturities=np.linspace(0.25,2.0,8),
                    N_list=(3,5,7,10,15,20), Kpoly=10, fft_kwargs=None):
    # Build FFT reference once
    base = compute_surfaces(model, rate, S0, strikes, maturities, N=10, Kpoly=Kpoly, fft_kwargs=fft_kwargs)
    ref_price = base["prices_fft"]; ref_iv = base["iv_fft"]

    rms_price, rms_iv, labels = [], [], []
    for N in N_list:
        cur = compute_surfaces(model, rate, S0, strikes, maturities, N=N, Kpoly=Kpoly, fft_kwargs=fft_kwargs)
        mask = np.isfinite(cur["prices_ope"]) & np.isfinite(ref_price)
        rp = np.sqrt(np.mean((cur["prices_ope"][mask] - ref_price[mask])**2))
        mask_iv = np.isfinite(cur["iv_ope"]) & np.isfinite(ref_iv)
        ri = np.sqrt(np.mean((cur["iv_ope"][mask_iv] - ref_iv[mask_iv])**2))
        rms_price.append(rp); rms_iv.append(ri); labels.append(N)

    plt.figure(figsize=(7,4))
    plt.plot(labels, rms_price, "o-", label="RMSE price")
    plt.plot(labels, rms_iv, "s-", label="RMSE IV")
    plt.xlabel("OPE degree N"); plt.ylabel("RMSE"); plt.title("OPE Convergence vs FFT")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig("sweep_ope_rmse.png")
# ---------------------------
# Utilities: Black–Scholes
# ---------------------------
def _norm_cdf(x):
    # Abramowitz-Stegun approximation for speed/robustness (no scipy)
    # or fallback to numpy's erf
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

def bs_call_price(S0, K, r, T, sigma):
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 * np.exp(-0.0) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)

def implied_vol_call(price, S0, K, r, T, lo=1e-8, hi=5.0, tol=1e-8, maxit=100):
    """Robust bisection IV for a call. Assumes price is arbitrage-consistent."""
    # Clamp to intrinsic / upper bound to be safe
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    # Very high vol cap price (hi) used as upper bound
    p_lo = bs_call_price(S0, K, r, T, lo)
    p_hi = bs_call_price(S0, K, r, T, hi)
    # If out of bounds numerically, project
    target = min(max(price, intrinsic), p_hi)
    a, b = lo, hi
    for _ in range(maxit):
        mid = 0.5 * (a + b)
        p_mid = bs_call_price(S0, K, r, T, mid)
        if abs(p_mid - target) < tol:
            return float(mid)
        if p_mid > target:
            b = mid
        else:
            a = mid
    return float(0.5 * (a + b))

# --------------------------------------------------------
# Library helpers
# --------------------------------------------------------

def get_ope_class(N):
    """Map integer N to the bound OPE class in the module."""
    mapping = {
        3:  opoe.OPEOptionPricerN3,
        5:  opoe.OPEOptionPricerN5,
        7:  opoe.OPEOptionPricerN7,
        9:  opoe.OPEOptionPricerN9,
        10: opoe.OPEOptionPricerN10,
        15: opoe.OPEOptionPricerN15,
        20: opoe.OPEOptionPricerN20,
        25: opoe.OPEOptionPricerN25,
        30: opoe.OPEOptionPricerN30,
    }
    if N not in mapping:
        raise ValueError(f"OPE degree N={N} not bound in module.")
    return mapping[N]

# --------------------------------------------------------
# Baseline comparison (MC / FFT / OPE-Direct)
# --------------------------------------------------------
def price_with_all_methods(strike, maturity, rate, model):
    """Helper to price a European call option with all pricers, recording time."""
    payoff = opoe.EuropeanCallPayoff(strike)

    results = {}

    # Monte Carlo
    t0 = time.perf_counter()
    MC = opoe.MCOptionPricer(maturity, rate, payoff, model,
                             solver_type=opoe.SolverType.Milstein,
                             num_paths=10000, num_steps=200)
    results["Monte Carlo"] = (MC.price(), time.perf_counter() - t0)

    # FFT pricer
    if isinstance(model, (opoe.GeometricBrownianMotionModel, opoe.HestonModel)):
        t0 = time.perf_counter()
        FFT = opoe.FFTOptionPricer(maturity, rate, payoff, model,
                                   Npow=10, A=1000)
        results["FFT"] = (FFT.price(), time.perf_counter() - t0)

    # OPE Direct
    t0 = time.perf_counter()
    OPE_direct = opoe.OPEOptionPricerN5(maturity, rate, payoff, model,
                                        K=5, solving_param=opoe.OPEMethod.Direct)
    results["OPE-Direct"] = (OPE_direct.price(), time.perf_counter() - t0)

    # OPE Integration
    t0 = time.perf_counter()
    OPE_integ = opoe.OPEOptionPricerN5(maturity, rate, payoff, model,
                                       K=5, solving_param=opoe.OPEMethod.Integration)
    results["OPE-Integration"] = (OPE_integ.price(), time.perf_counter() - t0)

    return results


def run_experiment():
    rate = 0.05
    maturity = 1.0
    strike = 100.0
    x0 = np.array([0.04, np.log(100.0)])
    heston = opoe.HestonModel(0.04, 2.0, 0.10, 0.3, -0.7, x0)

    results = []
    raw = price_with_all_methods(strike, maturity, rate, heston)
    for method, (price, runtime) in raw.items():
        results.append({"Method": method,
                        "Strike": strike,
                        "Maturity": maturity,
                        "Price": price,
                        "Runtime (s)": runtime})

    df = pd.DataFrame(results)
    return df



def plot_strike_sweep(model, rate, maturity):
    strikes = np.linspace(80, 120, 10)
    methods = ["Monte Carlo", "FFT", "OPE-Direct"]
    results = {m: [] for m in methods}

    for K in strikes:
        priced = price_with_all_methods(K, maturity, rate, model)
        for m in methods:
            if m in priced:
                results[m].append(priced[m][0])
            else:
                results[m].append(np.nan)

    plt.figure(figsize=(8, 5))
    for m, values in results.items():
        plt.plot(strikes, values, marker="o", label=m)
    plt.title("Option price vs. Strike (Heston)")
    plt.xlabel("Strike")
    plt.ylabel("Call Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("strikesweep.png")

def plot_maturity_sweep(model, rate, strike):
    maturities = np.linspace(0.1, 2.0, 10)
    methods = ["Monte Carlo", "FFT", "OPE-Direct"]
    results = {m: [] for m in methods}

    for T in maturities:
        priced = price_with_all_methods(strike, T, rate, model)
        for m in methods:
            if m in priced:
                results[m].append(priced[m][0])
            else:
                results[m].append(np.nan)
                
                

    plt.figure(figsize=(8, 5))
    for m, values in results.items():
        plt.plot(maturities, values, marker="s", label=m)
    plt.title("Option price vs. Maturity (Heston)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Call Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("matsweep.png")

def compare_ope_methods(model, rate, strike, maturity, N=5, K=5):
    payoff = opoe.EuropeanCallPayoff(strike)
    OPEClass = get_ope_class(N)

    pricer_direct = OPEClass(maturity, rate, payoff, model, K=K,
                             solving_param=opoe.OPEMethod.Direct)
    pricer_integ  = OPEClass(maturity, rate, payoff, model, K=K,
                             solving_param=opoe.OPEMethod.Integration)

    p_direct = pricer_direct.price()
    p_integ  = pricer_integ.price()
    return p_direct, p_integ

def plot_ope_direct_vs_integration(model, rate, strike, maturity, N_list=(3,5,7,10), K_list=(3,5,7,10)):
    rows = []
    for N in N_list:
        for K in K_list:
            try:
                pD, pI = compare_ope_methods(model, rate, strike, maturity, N=N, K=K)
                rows.append({"N": N, "K": K, "OPE-Direct": pD, "OPE-Integration": pI})
            except Exception as e:
                rows.append({"N": N, "K": K, "OPE-Direct": np.nan, "OPE-Integration": np.nan})

    df = pd.DataFrame(rows).sort_values(["N", "K"])
    print("\n=== OPE Direct vs Integration (table) ===\n")
    print(df)

    # Plot bars grouped by (N,K)
    labels = [f"N={n},K={k}" for n, k in zip(df["N"], df["K"])]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, df["OPE-Direct"], width, label="Direct")
    plt.bar(x + width/2, df["OPE-Integration"], width, label="Integration")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(f"OPE: Direct vs Integration @ Strike={strike}, T={maturity}")
    plt.ylabel("Call Price")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ope_direct_vs_integration.png")

    # Also heatmaps over N and K (optional/handy)
    try:
        pivotD = df.pivot(index="N", columns="K", values="OPE-Direct")
        pivotI = df.pivot(index="N", columns="K", values="OPE-Integration")

        plt.figure(figsize=(8, 4))
        plt.imshow(pivotD, aspect="auto")
        plt.title("OPE-Direct price heatmap")
        plt.xlabel("K")
        plt.ylabel("N")
        plt.colorbar(label="Price")
        plt.xticks(range(len(pivotD.columns)), pivotD.columns)
        plt.yticks(range(len(pivotD.index)), pivotD.index)
        plt.tight_layout()
        plt.savefig("ope_direct_heatmap.png")

        plt.figure(figsize=(8, 4))
        plt.imshow(pivotI, aspect="auto")
        plt.title("OPE-Integration price heatmap")
        plt.xlabel("K")
        plt.ylabel("N")
        plt.colorbar(label="Price")
        plt.xticks(range(len(pivotI.columns)), pivotI.columns)
        plt.yticks(range(len(pivotI.index)), pivotI.index)
        plt.tight_layout()
        plt.savefig("ope_integration_heatmap.png")
    except Exception:
        # If pivot fails due to missing combos, just skip heatmaps
        pass

def plot_surface_comparison(model, rate, S0=100.0,
                            strikes=np.linspace(80, 120, 9),
                            maturities=np.linspace(0.25, 2.0, 8),
                            N=10, K=10):
    """
    Compare volatility surfaces from FFT vs OPE-Direct.
    Produces side-by-side surfaces and a difference plot.
    """
    methods = ["FFT", "OPE-Direct"]
    surfaces = {m: np.zeros((len(maturities), len(strikes))) for m in methods}
    payoff = opoe.EuropeanCallPayoff(float(100))
    OPEClass = get_ope_class(N)
    for iT, T in enumerate(maturities):
        FFTpricer = opoe.FFTOptionPricer(T, rate, payoff, model, Npow=10, A=1000)
        OPE_pricer = OPEClass(T, rate, payoff, model, K=K, solving_param=opoe.OPEMethod.Direct)


        for iK, K_strike in enumerate(strikes):
            payoff = opoe.EuropeanCallPayoff(float(K_strike))
            for method in methods:
                try:
                    if method == "FFT":
                        p = FFTpricer.price()
                    elif method == "OPE-Direct":
        
                        p = OPE_pricer.price()
                    else:
                        p = np.nan
                    iv = implied_vol_call(p, S0, K_strike, rate, T)
                    surfaces[method][iT, iK] = iv
                except Exception:
                    surfaces[method][iT, iK] = np.nan

    X, Y = np.meshgrid(strikes, maturities)
    Zfft, Zope = surfaces["FFT"], surfaces["OPE-Direct"]

    # --- Side-by-side comparison ---
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X, Y, Zfft, cmap="viridis", edgecolor="none", alpha=0.85)
    ax1.set_title("Implied Vol Surface (FFT)")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Maturity")
    ax1.set_zlabel("Implied Vol")
    fig.colorbar(surf1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X, Y, Zope, cmap="plasma", edgecolor="none", alpha=0.85)
    ax2.set_title(f"Implied Vol Surface (OPE-Direct, N={N}, K={K})")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_zlabel("Implied Vol")
    fig.colorbar(surf2, ax=ax2, shrink=0.6)

    plt.tight_layout()
    plt.savefig("vol_surface_fft_vs_ope.png")

    # --- Difference surface (OPE - FFT) ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    diff = Zope - Zfft
    surf = ax.plot_surface(X, Y, diff, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax.set_title(f"Difference Surface (OPE - FFT, N={N}, K={K})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Vol Difference")
    fig.colorbar(surf, shrink=0.6)
    plt.tight_layout()
    plt.savefig("vol_surface_difference.png")

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    # 1) Table: MC / FFT / OPE-Direct
    df = run_experiment()
    print("\n=== Pricing Results (MC/FFT/OPE-Direct) ===\n")
    print(df)

    # 2) Heston setup
    x0 = np.array([0.04, np.log(100.0)])
    heston = opoe.HestonModel(0.04, 2.0, 0.10, 0.3, -0.7, x0)

    # 3) Sweeps
    plot_strike_sweep(heston, rate=0.05, maturity=1.0)
    plot_maturity_sweep(heston, rate=0.05, strike=100.0)

    # 4) OPE Direct vs Integration
    plot_ope_direct_vs_integration(
        heston, rate=0.05, strike=100.0, maturity=1.0,
        N_list=(3, 5, 7, 10), K_list=(3, 5, 7, 10)
    )

    # 5) Build surfaces once and run all diagnostics
    strikes = np.linspace(80, 120, 21)
    maturities = np.linspace(0.25, 2.0, 16)
    surf = compute_surfaces(heston, rate=0.05, S0=100.0,
                            strikes=strikes, maturities=maturities,
                            N=10, Kpoly=10,
                            fft_kwargs=dict(Npow=10, A=1000))

    # Save a fixed side-by-side + difference
    plot_surface_comparison_fixed(surf, S0=100.0, N=10, Kpoly=10)

    # Slices and total variance
    plot_smile_slices(surf, T_list=(0.25, 1.0, 2.0))
    plot_term_slices(surf, K_list=(85, 100, 115))
    plot_total_variance_heatmaps(surf)

    # Error maps and scatter
    plot_error_maps(surf, S0=100.0, r=0.05)

    # Arbitrage diagnostics
    arbitrage_diagnostics(surf, r=0.05)

    # OPE convergence vs FFT
    sweep_ope_order(heston, rate=0.05, S0=100.0,
                    strikes=strikes, maturities=maturities,
                    N_list=(3,5,7,10), Kpoly=10,
                    fft_kwargs=dict(Npow=10, A=1000))

if __name__ == "__main__":
    main()