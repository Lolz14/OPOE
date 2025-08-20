# app.py
import numpy as np
import sdefin


# Shared for Payoff, GBM Parameters to remove put does not work
def main():
    # ------------------------
    # 1. Define model parameters
    # ------------------------
    v0 = 0.04       # initial variance
    kappa = 2.0     # mean reversion speed
    theta = 0.1    # long-term variance
    sigma = 0.3     # vol of vol
    rho = -0.7      # correlation
    x0 = np.array([0.20, np.log(100.0)])

    # ------------------------
    # 2. Define payoff
    # ------------------------
    strike = 100.0
    payoff = sdefin.EuropeanCallPayoff(strike)
    payoff2 = sdefin.EuropeanCallPayoff(strike)
    payoff3 = sdefin.EuropeanCallPayoff(strike)
 


    heston = sdefin.HestonModel(0.04, 2, 0.10, 0.3, -0.7, x0)
    geometric_brownian_motion = sdefin.GeometricBrownianMotionModel(0.04, 0.2, 4.5)
    jacobi = sdefin.JacobiModel(0.04, 1.5, 0.09, 0.3, -0.7, 0.0, 1.0, x0)
           

    # ------------------------
    # 3. Define solvers
    # ------------------------

    # call the helper
    paths = sdefin.EulerMaruyamaSolver(heston)
    
    fftpricer = sdefin.FFTOptionPricer(1, 0.05, payoff2, heston, 10, 10)
    
    print("FFT Option price:", fftpricer.price())
                
    
    OPEpricer = sdefin.OPEOptionPricerN5(1, 0.05, payoff, heston,  paths.solve, sdefin.QuadratureMethod.QAGI, 100)
    
    print("Option price:", OPEpricer.price())
    


 


if __name__ == "__main__":
    main()
