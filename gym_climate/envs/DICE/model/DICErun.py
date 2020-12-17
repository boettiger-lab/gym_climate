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

    def integrate(self, action, t):
        #TT = np.linspace(2000, 2500, 100, dtype = np.int32)
        self.t = t
        super().__init__(t=self.t)

        self.InitializeLabor(self.l, self.t)
        self.InitializeTFP(self.al, self.t)
        self.InitializeGrowthSigma(self.gsig, self.t)
        self.InitializeSigma(self.sigma, self.gsig, self.cost1, self.t)
        self.InitializeCarbonTree(self.cumetree, self.t)

        utility = self.fOBJ(action, 1.0, self.I, self.K, self.al, self.l,\
            self.YGROSS, self.sigma, self.EIND, self.E, self.CCA, self.CCATOT,\
            self.cumetree, self.MAT, self.MU, self.ML, self.FORC, self.TATM, \
            self.TOCEAN, self.DAMFRAC, self.DAMAGES, self.ABATECOST, self.cost1,\
            self.MCABATE, self.CPRICE, self.YNET, self.Y, self.C, self.CPC,\
            self.PERIODU, self.CEMUTOTPER, self.RI, self.t)
        
        return utility
