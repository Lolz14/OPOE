# app.py
import numpy as np
import sdefin


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
    payoff = sdefin.EuropeanPutPayoff(strike)
    payoff2 = sdefin.EuropeanCallPayoff(strike + 50)
    payoff3 = sdefin.EuropeanCallPayoff(strike)
 


    heston = sdefin.HestonModel(0.04, 2, 0.10, 0.3, -0.7, x0)
    geometric_brownian_motion = sdefin.GeometricBrownianMotionModel(0.05, 0.2, 4.6)
    jacobi = sdefin.JacobiModel(0.04, 1.5, 0.09, 0.3, -0.7, 0.0, 1.0, x0)
           

    # ------------------------
    # 3. Define solvers
    # ------------------------

    # call the helper
    
    fftpricer = sdefin.FFTOptionPricer(1, 0.05, payoff, heston, 5, 100)
    
    print("FFT Option price:", fftpricer.price())
            
    
    OPEpricer = sdefin.OPEOptionPricerN7(1, 0.05, payoff, heston,  sdefin.SolverType.IJK, sdefin.QuadratureMethod.TanhSinh, 100)
    
    print("Option price:", OPEpricer.price())
        
    MC = sdefin.MCOptionPricer(1, 0.05, payoff, heston,   sdefin.SolverType.EulerMaruyama )
    
    
    print("MC price:", MC.price())



 


if __name__ == "__main__":
    main()
