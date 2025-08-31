[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
# opoe — Option Pricing with Orthogonal Expansions

A C++20 library with Python bindings for fast and accurate option pricing via Orthogonal Polynomial Expansions (OPE), alongside FFT-, Monte Carlo-, and closed‑form methods.

## Introduction: the OPE method (what, why, how)

- **What**: The OPE method prices options by expanding the log-price density around an auxiliary density w(x) (typically a Gaussian or a finite Gaussian mixture), building an orthonormal polynomial basis {Hn} with respect to w, and evaluating a truncated series π(N) = Σn≤N fn ℓn where fn are payoff projections and ℓn are model-implied moments.

- **Why**: For many stochastic volatility models (Heston, Stein–Stein, Hull–White, Jacobi), log-price distributions are close to a Gaussian mixture. With the right auxiliary density, the expansion converges quickly, achieving high accuracy at low polynomial orders with excellent performance.

- **How**:
  1. Choose w(x): a single Gaussian or, more robustly, a finite Gaussian mixture approximating the log-price density.
  2. Build the orthonormal basis (ONB) via a stable three-term recurrence derived from mixture components.
  3. Compute model moments ℓn using the polynomial property (matrix exponential of the generator acting on a polynomial basis).
  4. Compute payoff projections fn either (i) in closed form for standard payoffs and Gaussian components (Direct method) or (ii) with 1D quadrature against w (Integration method).
  5. Form the truncated price π(N) = Σn=0..N fn ℓn and discount/adjust (e.g., put–call parity).

Key highlights:
- Finite Gaussian mixtures better match tails and skew than a single Gaussian, improving stability and speed of convergence.
- Moments under polynomial diffusions reduce to linear algebra (matrix exponentials) on a finite polynomial space.
- Practical implementations combine robust mixture construction (via weighted paths) with stable recurrence algorithms to build the ONB.

## Features

- C++20 core with modular namespaces: polynomials, stats (densities/mixtures), quadrature (Boost tanh–sinh, GSL), sde (GBM, Heston, Stein–Stein, Hull–White, Jacobi), options (CF/MC/FFT/OPE pricers).
- Python extension module via pybind11 and scikit-build-core.
- Multiple pricing engines:
  - Closed-form (Black–Scholes)
  - Monte Carlo (Euler–Maruyama, Milstein, IJK)
  - FFT (Carr–Madan)
  - OPE (Direct and Integration)
  - 
## Roadmap

This project is under active development. Planned features and refinements include:
- Extended payoff support (barrier, Asian, lookback options)
- GPU acceleration for moment computation
- Cross-platform prebuilt binaries (PyPI/Conda)
- Calibration module for each pricer
- Data retrieval via API

Contributions, suggestions, and collaborations are very welcome!

## Prerequisites

- **Python**: 3.9+
- **C++ toolchain**: C++20-capable compiler (GCC/Clang/MSVC), CMake ≥ 3.25, Ninja
- **Python build deps (PEP 517)**: scikit-build-core, pybind11, numpy, pandas, matplotlib
- **System libraries**:
  - FFTW3 (double-precision)
  - GSL (GNU Scientific Library)
  - Eigen3 (headers)
  - Boost headers
  - OpenMP
- **Helpers**: pkg-config (recommended on Linux/macOS to find FFTW/GSL)

The build will use CMake’s find_package for Eigen3/Boost, and a config or pkg-config path for FFTW3 and GSL. If config packages are not present, pkg-config is used.

## Installation

You can install via pip from the project root (editable or regular). Ensure the system libraries above are installed first.

### Linux (Ubuntu/Debian)

```bash
# System toolchain and libs
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build pkg-config python3-dev python3-venv libfftw3-dev libgsl-dev libeigen3-dev libboost-dev

# Optional: OpenMP (usually included with GCC)
# sudo apt-get install -y libomp-dev

# Python build deps (in a venv is recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install .
# or for dev
# pip install -e .
```

### macOS (Homebrew)

```bash
# Xcode CLT
xcode-select --install

# Core libs
brew install cmake ninja pkg-config fftw gsl eigen boost

# Python build deps
python -m venv .venv && source .venv/bin/activate
pip install .
# or dev
# pip install -e .
```

### Windows

Option A — vcpkg (recommended for native MSVC builds)
```powershell
# Prereqs: Visual Studio 2022 Build Tools (MSVC v143), CMake, Ninja
# Install vcpkg and integrate
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

# Install libraries (x64)
C:\vcpkg\vcpkg.exe install gsl fftw3 eigen3 boost --triplet x64-windows

# Build with pip, pointing to vcpkg toolchain
python -m venv .venv
.\.venv\Scripts\activate
pip install -v . --config-settings=cmake.define.CMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
# or dev
# pip install -v -e . --config-settings=cmake.define.CMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Option B — conda (cross‑platform)
```bash
conda create -n opoe python=3.11
conda activate opoe
conda install -c conda-forge cmake ninja pybind11 scikit-build-core numpy \
    gsl fftw eigen boost-cpp
pip install .
# or dev
# pip install -e .
```
- If GSL/FFTW are installed in nonstandard locations, set `CMAKE_PREFIX_PATH` or ensure `pkg-config` can find them. 
- On Windows with vcpkg, using `CMAKE_TOOLCHAIN_FILE` is essential so CMake resolves dependencies correctly.

### Known Issues:
- MSVC with C++20 can fail when compiling Eigen (see [Eigen issues](https://gitlab.com/libeigen/eigen/-/issues?label_name=3.4)).
  Workarounds:
  - Use WSL + GCC/Clang (recommended)
  - Downgrade to C++17 on MSVC (requires strong code refactoring, eliminating every C++20 feature used in the code)

  
## Quick start (Python)

The bindings expose models, payoffs, and pricers with a uniform interface:
- `Pricer(ttm, rate, payoff, model, ...)` and `price()`
- Payoffs include European call/put
- Models include GBM, Heston, Stein–Stein, Hull–White, Jacobi
- Pricers include Closed-form (CF), Monte Carlo (MC), FFT, and OPE

Example (illustrative):
```python
import opoe as op

# Model (e.g., Heston) and payoff
model = op.HestonModel(v0=0.04, kappa=2.0, theta=0.10, sigma_v=0.3, rho=-0.7, x0=0.0)  # x0 = log(S0)
payoff = op.EuropeanCallPayoff(strike=100.0)

# OPE Pricer (Direct method)
pricer = op.OPEOptionPricer(
    ttm=1.0, rate=0.05,
    payoff=payoff, model=model,
    mixture_components=10,              # K
    solving_method=op.OPEMethod.Direct, # or Integration
    solver=op.SolverType.Milstein,      # SDE path solver for mixture construction
)
print("OPE price:", pricer.price())

# Alternatives: FFT or Monte Carlo
# fft_pricer = op.FFTOptionPricer(ttm=1.0, rate=0.05, payoff=payoff, model=model, n_pow=10, A=1000)
# mc_pricer = op.MCOptionPricer(ttm=1.0, rate=0.05, payoff=payoff, model=model,
#                               solver=op.SolverType.Milstein, paths=10000, steps=200)
```

Tip:
- For Black–Scholes/GBM, use the closed-form pricer exposed in the module.


## Testing
By running the command below:
```bash
python app.py #python3 when on Linux 
```

A test-run example will be run showcasing capabilities of the library and comparing the OPE method to others.

## Documentation

This repository does not ship a preconfigured documentation site yet. The doxygen documentation is available through:


```bash
# Install doxygen (Linux: apt/brew; Windows: choco/scoop or installer)
doxygen Doxyfile
# Open doc/html/index.html
```


## How to cite

Please cite both the library and the methodology.

- Library: [CITATION](./CITATION.cff) file.

- Methodology:
  - Ackerer, D., & Filipović, D. (2019). Option pricing with orthogonal polynomial expansions.

## License

This project is distributed under the **MIT License**. See the [LICENSE](./LICENSE) file for the full text.
