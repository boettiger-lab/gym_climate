import numpy as np
from gym_climate.envs.DICE.model.DICEfunctions import *
class DICE(DICEfunctions):
    def __init__(self, t=np.array([1]), duration=100):
        super().__init__(t=np.array([1]), duration=100)
        self.K = np.zeros(self.NT)
        self.YGROSS = np.zeros(self.NT)
        self.EIND = np.zeros(self.NT)
        self.E = np.zeros(self.NT)
        self.CCA = np.zeros(self.NT)
        self.CCATOT = np.zeros(self.NT)
        self.MAT = np.zeros(self.NT)
        self.ML = np.zeros(self.NT)
        self.MU = np.zeros(self.NT)
        self.FORC = np.zeros(self.NT)
        self.TATM = np.zeros(self.NT)
        self.TOCEAN = np.zeros(self.NT)
        self.DAMFRAC = np.zeros(self.NT)
        self.DAMAGES = np.zeros(self.NT)
        self.ABATECOST = np.zeros(self.NT)
        self.MCABATE = np.zeros(self.NT)
        self.CPRICE = np.zeros(self.NT)
        self.YNET = np.zeros(self.NT)
        self.Y = np.zeros(self.NT)
        self.I = np.zeros(self.NT)
        self.C = np.zeros(self.NT)
        self.CPC = np.zeros(self.NT)
        self.RI = np.zeros(self.NT)
        self.PERIODU = np.zeros(self.NT)
        self.CEMUTOTPER = np.zeros(self.NT)
        self.test_flag = False

    def integrate(self, action, t):
        #TT = np.linspace(2000, 2500, 100, dtype = np.int32)
        self.t = t
        # Update the dynamic parameters
        super().__init__(t=self.t)
        self.InitializeLabor(self.l, self.t)
        self.InitializeTFP(self.al, self.t)
        self.InitializeGrowthSigma(self.gsig, self.t)
        self.InitializeSigma(self.sigma, self.gsig, self.cost1, self.t)
        self.InitializeCarbonTree(self.cumetree, self.t)

        # Evaluate the utility function
        utility = self.fOBJ(action, 1.0, self.I, self.K, self.al, self.l,\
            self.YGROSS, self.sigma, self.EIND, self.E, self.CCA, self.CCATOT,\
            self.cumetree, self.MAT, self.MU, self.ML, self.FORC, self.TATM, \
            self.TOCEAN, self.DAMFRAC, self.DAMAGES, self.ABATECOST, self.cost1,\
            self.MCABATE, self.CPRICE, self.YNET, self.Y, self.C, self.CPC,\
            self.PERIODU, self.CEMUTOTPER, self.RI, self.t, self.test_flag)

        return utility

    def get_obs(self, t):
        # Returns all the observables. 
        # Definitely some room here to reduce the observation space size.
        return np.array([self.I[t], self.K[t], self.al[t], self.l[t],\
            self.YGROSS[t], self.sigma[t], self.EIND[t], self.E[t], self.CCA[t], self.CCATOT[t],\
            self.cumetree[t], self.MAT[t], self.MU[t], self.ML[t], self.FORC[t], self.TATM[t], \
            self.TOCEAN[t], self.DAMFRAC[t], self.DAMAGES[t], self.ABATECOST[t], self.cost1[t],\
            self.MCABATE[t], self.CPRICE[t], self.YNET[t], self.Y[t], self.C[t], self.CPC[t],\
            self.PERIODU[t], self.CEMUTOTPER[t]])
