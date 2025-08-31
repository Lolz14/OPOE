import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opoe
import math
import time
from mpl_toolkits.mplot3d import Axes3D  

# ---------------------------
# Utilities: Black–Scholes
# ---------------------------
def _norm_cdf(x):
    # Abramowitz-Stegun approximation for speed
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

#Did not use CFOptionPricer for clarity
def bs_call_price(S0, K, r, T, sigma):
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 * np.exp(-0.0) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)

def implied_vol_call(price, S0, K, r, T, lo=1e-8, hi=5.0, tol=1e-8, maxit=100):
    """Bisection IV for a call."""
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

    # Also heatmaps over N and K
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
    #1) Table: MC / FFT / OPE-Direct
    df = run_experiment()
    print("\n=== Pricing Results (MC/FFT/OPE-Direct) ===\n")
    print(df)

    #2) Heston setup
    x0 = np.array([0.04, np.log(100.0)])
    heston = opoe.HestonModel(0.04, 2.0, 0.10, 0.3, -0.7, x0)
 

    #3) Sweeps
    plot_strike_sweep(heston, rate=0.05, maturity=1.0)
    plot_maturity_sweep(heston, rate=0.05, strike=100.0)

    #4) OPE Direct vs Integration
    plot_ope_direct_vs_integration(
        heston, rate=0.05, strike=100.0, maturity=1.0,
        N_list=(3, 5, 7, 10), K_list=(3, 5, 7, 10)
    )

    #5) Build surfaces once and run all diagnostics
    strikes = np.linspace(80, 120, 21)
    maturities = np.linspace(0.25, 2.0, 16)
    surf = plot_surface_comparison(heston, rate=0.05, S0=100.0,
                            strikes=strikes, maturities=maturities,
                            N=10, K=10)


if __name__ == "__main__":
    main()