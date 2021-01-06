from numba import njit,guvectorize,float64
import numpy as np
from gym_climate.envs.DICE.model.DICEparams import *
import pdb
import csv

class DICEfunctions(DICEparams):
    def __init__(self, t, duration=100):
        super().__init__(t, duration=100)

    def InitializeLabor(self, il, iNT):
        il[iNT] = il[iNT-1]*(self.popasym / il[iNT-1])**self.popadj

    def InitializeTFP(self, ial, iNT):
        ial[iNT] = ial[iNT-1]/(1-self.ga[0])

    def InitializeGrowthSigma(self, igsig, iNT):
        igsig[iNT] = igsig[iNT-1]*((1+self.dsig)**self.tstep)

    def InitializeSigma(self, isigma, igsig, icost1, iNT):
        isigma[iNT] =  isigma[iNT-1] * np.exp(igsig[iNT-1] * self.tstep)
        icost1[iNT] = self.pbacktime[1] * isigma[iNT]  / self.expcost2 /1000

    def InitializeCarbonTree(self, icumetree, iNT):
        icumetree[iNT] = icumetree[iNT-1] + self.etree[0]*(5/3.666)

    """
    Functions of the model
    """

    """
    First: Functions related to emissions of carbon and weather damages
    """

    # Retuns the total carbon emissions; Eq. 18
    def fE(self, iEIND, index):
        flag=1
        if index == 0:
            flag = 0
        return iEIND[index] + self.etree[flag]

    #Eq.14: Determines the emission of carbon by industry EIND
    def fEIND(self, iYGROSS, iMIU, isigma, index):
        return isigma[index] * iYGROSS[index] * (1 - iMIU)

    #Cumulative industrial emission of carbon
    def fCCA(self, iCCA, iEIND, index):
        return iCCA[index-1] + iEIND[index-1] * 5 / 3.666

    #Cumulative total carbon emission
    def fCCATOT(self, iCCA, icumetree, index):
        return iCCA[index] + icumetree[index]

    #Eq. 22: the dynamics of the radiative forcing
    def fFORC(self, iMAT, index):
        return self.fco22x * np.log(iMAT[index]/588.000)/np.log(2) + self.forcoth[index]

    # Dynamics of Omega; Eq.9
    def fDAMFRAC(self, iTATM, index):
        return self.a1*iTATM[index] + self.a2*iTATM[index]**self.a3

    #Calculate damages as a function of Gross industrial production; Eq.8 
    def fDAMAGES(self, iYGROSS, iDAMFRAC, index):
        return iYGROSS[index] * iDAMFRAC[index]

    #Dynamics of Lambda; Eq. 10 - cost of the reudction of carbon emission (Abatement cost)
    def fABATECOST(self, iYGROSS, iMIU, icost1, index):
        return iYGROSS[index] * icost1[index] * iMIU**self.expcost2

    #Marginal Abatement cost
    def fMCABATE(self, iMIU, index):
        flag=1
        if index == 0:
            flag = 0
        return self.pbacktime[flag] * iMIU**(self.expcost2-1)

    #Price of carbon reduction
    def fCPRICE(self, iMIU, index):
        flag=1
        if index == 0:
            flag = 0
        return self.pbacktime[flag] * (iMIU)**(self.expcost2-1)

    #Eq. 19: Dynamics of the carbon concentration in the atmosphere 
    def fMAT(self, iMAT, iMU, iE, index):
        if(index == 0):
            return self.mat0
        else:
            return iMAT[index-1]*self.b11 + iMU[index-1]*self.b21 + iE[index-1] * 5 / 3.666

    #Eq. 21: Dynamics of the carbon concentration in the ocean LOW level
    def fML(self, iML, iMU, index):
        if(index == 0):
            return self.ml0
        else:
            return iML[index-1] * self.b33  + iMU[index-1] * self.b23

    #Eq. 20: Dynamics of the carbon concentration in the ocean UP level
    def fMU(self, iMAT, iMU, iML, index):
        if(index == 0):
            return self.mu0
        else:
            return iMAT[index-1]*self.b12 + iMU[index-1]*self.b22 + iML[index-1]*self.b32

    #Eq. 23: Dynamics of the atmospheric temperature
    def fTATM(self, iTATM, iFORC, iTOCEAN, index):
        if(index == 0):
            return self.tatm0
        else:
            return iTATM[index-1] + self.c1 * (iFORC[index] - (self.fco22x/self.t2xco2) * iTATM[index-1] - self.c3 * (iTATM[index-1] - iTOCEAN[index-1]))

    #Eq. 24: Dynamics of the ocean temperature
    def fTOCEAN(self, iTATM, iTOCEAN, index):
        if(index == 0):
            return self.tocean0
        else:
            return iTOCEAN[index-1] + self.c4 * (iTATM[index-1] - iTOCEAN[index-1])

    """
    Second: Function related to economic variables
    """

    #The total production without climate losses denoted previously by YGROSS
    def fYGROSS(self, ial, il, iK, index):
        return ial[index] * ((il[index]/1000)**(1-self.gama)) * iK[index]**self.gama

    #The production under the climate damages cost
    def fYNET(self, iYGROSS, iDAMFRAC, index):
        return iYGROSS[index] * (1 - iDAMFRAC[index])

    #Production after abatement cost
    def fY(self, iYNET, iABATECOST, index):
        return iYNET[index] - iABATECOST[index]

    #Consumption Eq. 11
    def fC(self, iY, iI, index):
        return iY[index] - iI[index]

    #Per capita consumption, Eq. 12
    def fCPC(self, iC, il, index):
        return 1000 * iC[index] / il[index]

    #Saving policy: investment
    def fI(self, iS, iY, index):
        return iS * iY[index] 

    #Capital dynamics Eq. 13
    def fK(self, iK, iI, index):
        if(index == 0):
            return self.k0
        else:
            return (1-self.dk)**self.tstep * iK[index-1] + self.tstep * iI[index-1]

    #Interest rate equation; Eq. 26 added in personal notes
    def fRI(self, iCPC, index):
        return (1 + self.prstp) * (iCPC[index+1]/iCPC[index])**(self.elasmu/self.tstep) - 1

    #Periodic utility: A form of Eq. 2
    def fCEMUTOTPER(self, iPERIODU, il, index):
        flag=1
        if index == 0:
            flag=0
        return iPERIODU[index] * il[index] * self.rr[flag]

    #The term between brackets in Eq. 2
    def fPERIODU(self, iC, il, index):
        return ((iC[index]*1000/il[index])**(1-self.elasmu) - 1) / (1 - self.elasmu) - 1

    #utility function
    def fUTILITY(self, iCEMUTOTPER, resUtility):
        resUtility[0] = self.tstep * self.scale1 * np.sum(iCEMUTOTPER) + self.scale2

    #The objective function
    #It returns the utility as scalar
    def fOBJ(self, action, sign, iI, iK, ial, il, iYGROSS, isigma, iEIND, iE, iCCA, iCCATOT, icumetree,\
             iMAT, iMU, iML, iFORC, iTATM, iTOCEAN, iDAMFRAC, iDAMAGES, iABATECOST, icost1,\
             iMCABATE, iCPRICE, iYNET, iY, iC, iCPC, iPERIODU, iCEMUTOTPER, iRI, iNT, test_flag):
        
        iMIU = action[0]
        iS = action[1]
        
        if iNT == 1:
            times = [0, 1]
        else:
            times = [iNT]


        for i in times:
            iK[i] = self.fK(iK,iI,i)
            iYGROSS[i] = self.fYGROSS(ial,il,iK,i)
            iEIND[i] = self.fEIND(iYGROSS, iMIU, isigma,i)
            iE[i] = self.fE(iEIND,i)
            iCCA[i] = self.fCCA(iCCA,iEIND,i)
            iCCATOT[i] = self.fCCATOT(iCCA,icumetree,i)
            iMAT[i] = self.fMAT(iMAT,iMU,iE,i)
            iML[i] = self.fML(iML,iMU,i)
            iMU[i] = self.fMU(iMAT,iMU,iML,i)
            iFORC[i] = self.fFORC(iMAT,i)
            iTATM[i] = self.fTATM(iTATM,iFORC,iTOCEAN,i)
            iTOCEAN[i] = self.fTOCEAN(iTATM,iTOCEAN,i)
            iDAMFRAC[i] = self.fDAMFRAC(iTATM,i)
            iDAMAGES[i] = self.fDAMAGES(iYGROSS,iDAMFRAC,i)
            iABATECOST[i] = self.fABATECOST(iYGROSS,iMIU,icost1,i)
            iMCABATE[i] = self.fMCABATE(iMIU,i)
            iCPRICE[i] = self.fCPRICE(iMIU,i)
            iYNET[i] = self.fYNET(iYGROSS, iDAMFRAC, i)
            iY[i] = self.fY(iYNET,iABATECOST,i)
            iI[i] = self.fI(iS,iY,i)
            iC[i] = self.fC(iY,iI,i)
            iCPC[i] = self.fCPC(iC,il,i)
            iPERIODU[i] = self.fPERIODU(iC,il,i)
            iCEMUTOTPER[i] = self.fCEMUTOTPER(iPERIODU,il,i)
            # Made some changes here to iRI from pyDICE to be defined recursively 
            # rather than at once
            if i > 0:
                iRI[i-1] = self.fRI(iCPC,i-1)

        # DEBUGGING SECTION
        if iNT == 99 and test_flag:
            myfile = open('parameter_check.csv')
            rows = []
            reader = csv.reader(myfile)
            for row in reader:
                rows.append(row[0].split('\t'))
            for i, parameter in enumerate([iK, iYGROSS, iEIND, iE, iCCA,\
                     iCCATOT, iMAT, iML, iMU, iFORC, iTATM, iTOCEAN,\
                     iDAMFRAC, iDAMAGES, iABATECOST, iMCABATE, iCPRICE, \
                     iYNET, iY, iI, iC, iCPC, iPERIODU, iCEMUTOTPER]):
                 test = [float("".join(x.replace('"', '').split(","))) for x in rows[i]]
                 x = list(map(lambda x: float(str(x)[:8]), list(parameter)))
                 y = list(map(lambda x: float(str(x)[:8]), list(test)))
                 assert y == x,  "Model error"
            myfile.close()
        resUtility = np.zeros(1)
        self.fUTILITY(iCEMUTOTPER, resUtility)
        
        return sign*resUtility[0]
