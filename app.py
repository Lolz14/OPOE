# app.py
import numpy as np
import opoe


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
    payoff = opoe.EuropeanCallPayoff(strike)
    payoff3 = opoe.EuropeanCallPayoff(strike)
 


    heston = opoe.HestonModel(0.04, 2, 0.10, 0.3, -0.7, x0)
    geometric_brownian_motion = opoe.GeometricBrownianMotionModel(0.05, 0.2, 4.6)
    jacobi = opoe.JacobiModel(0.04, 1.5, 0.09, 0.3, -0.7, 0.0, 1.0, x0)
           

    # ------------------------
    # 3. Define solvers
    # ------------------------

    # call the helper
    

            
    
    OPEpricer = opoe.OPEOptionPricerN5(1, 0.05, payoff3, heston, 5, opoe.OPEMethod.Direct)
    
    print("Option price:", OPEpricer.price())
        
    MC = opoe.FFTOptionPricer(1, 0.05, payoff3, heston, 10, 1000)
    
    
    print("MC price:", MC.price())



 


if __name__ == "__main__":
    main()
