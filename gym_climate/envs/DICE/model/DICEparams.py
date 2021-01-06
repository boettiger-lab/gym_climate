import numpy as np
from numba.experimental import jitclass
class DICEparams():
    def __init__(self, t, duration=100):
        #Set
        self.t = t
        self.NT = duration

        #Parameters

        self.fosslim = 6000 # Maximum cumulative extraction fossil fuels (GtC); denoted by CCum
        self.tstep  = 5 # Years per Period
        self.ifopt  = 0 # Indicator where optimized is 1 and base is 0

        #Preferences

        self.elasmu = 1.45 #  Elasticity of marginal utility of consumption
        self.prstp = 0.015 #   Initial rate of social time preference per year 

        #** Population and technology
        self.gama  = 0.300 #   Capital elasticity in production function         /.300 /
        self.pop0  = 7403   # Initial world population 2015 (millions)          /7403 /
        self.popadj = 0.134 #  Growth rate to calibrate to 2050 pop projection  /0.134/
        self.popasym = 11500 # Asymptotic population (millions)                 /11500/
        self.dk  = 0.100 #     Depreciation rate on capital (per year)           /.100 /
        self.q0  = 105.5 #     Initial world gross output 2015 (trill 2010 USD) /105.5/
        self.k0  = 223 #     Initial capital value 2015 (trill 2010 USD)        /223  /
        self.a0  = 5.115 #     Initial level of total factor productivity       /5.115/
        self.ga0  = 0.076 #    Initial growth rate for TFP per 5 years          /0.076/
        self.dela  = 0.005 #   Decline rate of TFP per 5 years                  /0.005/

        #** Emissions parameters
        self.gsigma1  = -0.0152 # Initial growth of sigma (per year)            /-0.0152/
        self.dsig  = -0.001 #   Decline rate of decarbonization (per period)    /-0.001 /
        self.eland0 = 2.6 #  Carbon emissions from land 2015 (GtCO2 per year)   / 2.6   /
        self.deland = 0.115 # Decline rate of land emissions (per period)        / .115  /
        self.e0 = 35.85 #    Industrial emissions 2015 (GtCO2 per year)       /35.85  /
        self.miu0  = 0.03 #   Initial emissions control rate for base case 2015  /.03    /

        #** Carbon cycle
        #* Initial Conditions
        self.mat0 = 851 #  Initial Concentration in atmosphere 2015 (GtC)       /851  /
        self.mu0  = 460 #  Initial Concentration in upper strata 2015 (GtC)     /460  /
        self.ml0  = 1740 #  Initial Concentration in lower strata 2015 (GtC)    /1740 /
        self.mateq = 588 # mateq Equilibrium concentration atmosphere  (GtC)    /588  /
        self.mueq  = 360 # mueq Equilibrium concentration in upper strata (GtC) /360  /
        self.mleq = 1720 # mleq Equilibrium concentration in lower strata (GtC) /1720 /

        #* Flow paramaters, denoted by Phi_ij in the model
        self.b12  = 0.12 #    Carbon cycle transition matrix                     /.12  /
        self.b23  = 0.007 #   Carbon cycle transition matrix                    /0.007/
        #* These are for declaration and are defined later
        self.b11  = None   # Carbon cycle transition matrix
        self.b21  = None  # Carbon cycle transition matrix
        self.b22  = None  # Carbon cycle transition matrix
        self.b32  = None  # Carbon cycle transition matrix
        self.b33  = None  # Carbon cycle transition matrix
        self.sig0  = None  # Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)

        #** Climate model parameters
        self.t2xco2  = 3.1 # Equilibrium temp impact (oC per doubling CO2)    / 3.1 /
        self.fex0  = 0.5 #   2015 forcings of non-CO2 GHG (Wm-2)              / 0.5 /
        self.fex1  = 1.0 #   2100 forcings of non-CO2 GHG (Wm-2)              / 1.0 /
        self.tocean0  = 0.0068 # Initial lower stratum temp change (C from 1900) /.0068/
        self.tatm0  = 0.85 #  Initial atmospheric temp change (C from 1900)    /0.85/
        self.c1  = 0.1005 #     Climate equation coefficient for upper level  /0.1005/
        self.c3  = 0.088 #     Transfer coefficient upper to lower stratum    /0.088/
        self.c4  = 0.025 #     Transfer coefficient for lower level           /0.025/
        self.fco22x  = 3.6813 # eta in the model; Eq.22 : Forcings of equilibrium CO2 doubling (Wm-2)   /3.6813 /

        #** Climate damage parameters
        self.a10  = 0 #     Initial damage intercept                         /0   /
        self.a20  = None #     Initial damage quadratic term
        self.a1  = 0 #      Damage intercept                                 /0   /
        self.a2  = 0.00236 #      Damage quadratic term                     /0.00236/
        self.a3  = 2.00 #      Damage exponent                              /2.00   /

        #** Abatement cost
        self.expcost2 = 2.6 # Theta2 Exponent of control cost function             / 2.6  /
        self.pback  = 550 #   Cost of backstop 2010$ per tCO2 2015          / 550  /
        self.gback  = 0.025 #   Initial cost decline backstop cost per period / .025/
        self.limmiu  = 1.2 #  Upper limit on control rate after 2150        / 1.2 /
        self.tnopol  = 45 #  Period before which no emissions controls base  / 45   /
        self.cprice0  = 2 # Initial base carbon price (2010$ per tCO2)      / 2    /
        self.gcprice  = 0.02 # Growth rate of base carbon price per year     /.02  /

        #** Scaling and inessential parameters
        #* Note that these are unnecessary for the calculations
        #* They ensure that MU of first period's consumption =1 and PV cons = PV utilty
        self.scale1  = 0.0302455265681763 #    Multiplicative scaling coefficient           /0.0302455265681763 /
        self.scale2  = -10993.704 #    Additive scaling coefficient       /-10993.704/;

        #* Parameters for long-run consistency of carbon cycle 
        #(Question)
        self.b11 = 1 - self.b12
        self.b21 = self.b12*self.mateq/self.mueq
        self.b22 = 1 - self.b21 - self.b23
        self.b32 = self.b23*self.mueq/self.mleq
        self.b33 = 1 - self.b32

        #* Further definitions of parameters
        self.a20 = self.a2
        self.sig0 = self.e0/(self.q0*(1-self.miu0))
        self.lam = self.fco22x/ self.t2xco2

        self.times = np.array([self.t-1, self.t]).reshape(-1)
        if self.t == 1:
            self.l = np.zeros(self.NT)
            self.l[0] = self.pop0 #Labor force
            self.al = np.zeros(self.NT)
            self.al[0] = self.a0
            self.gsig = np.zeros(self.NT)
            self.gsig[0] = self.gsigma1
            self.sigma = np.zeros(self.NT)
            self.sigma[0]= self.sig0
            self.cost1 = np.zeros(self.NT)
            self.cumetree = np.zeros(self.NT)
            self.cumetree[0] = 100
            self.forcoth = np.full(self.NT,self.fex0)
        self.ga = self.ga0 * np.exp(-self.dela*5*(self.times)) #TFP growth rate dynamics
        self.pbacktime = self.pback * (1-self.gback)**(self.times) #Backstop price
        self.etree = self.eland0*(1-self.deland)**(self.times) #Emissions from deforestration
        self.rr = 1/((1+self.prstp)**(self.tstep*(self.times)))
        #The following three equations define the exogenous radiative forcing  
        if self.t > 17:
            self.forcoth[self.t] = self.forcoth[self.t] + (self.fex1-self.fex0)
        else:
            self.forcoth[self.t] = self.forcoth[self.t] + (1/17)*(self.fex1-self.fex0)*(self.t)
        self.optlrsav = (self.dk + .004)/(self.dk + .004*self.elasmu + self.prstp)*self.gama #Optimal long-run savings rate used for transversality (Question)
        self.cpricebase = self.cprice0*(1+self.gcprice)**(5*(self.times))
