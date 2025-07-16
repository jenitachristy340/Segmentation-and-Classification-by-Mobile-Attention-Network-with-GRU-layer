from numpy.random import rand
from Global_Vars import Global_Vars
import numpy as np
from Model_MobileUnetPlusPlus import Model_MobileunetPlusPlus


def Obj_fun(Soln):
    Images = Global_Vars.Images
    GT = Global_Vars.GT
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 1:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i]).astype('uint8')
            Eval, Segimg = Model_MobileunetPlusPlus(Images, GT, sol)
            Fitn[i] = 1 / (Eval[4] + Eval[6])
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        Eval, Segimg = Model_MobileunetPlusPlus(Images, GT, sol)
        Fitn = 1 / (Eval[4] + Eval[6])
        return Fitn

