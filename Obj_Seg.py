from Global_Vars import Global_Vars
import numpy as np
from Model_AMNet_SSDV2 import Model_AMNet_SSDV2


def objfun_seg(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Pred, Eval = Model_AMNet_SSDV2(Feat, Target, sol=sol)
            Fitn[i] = 1 / (Eval[0, 5])  # 1 / IoU
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Pred, Eval = Model_AMNet_SSDV2(Feat, Target, sol=sol)
        Fitn = 1 / (Eval[0, 5])  # 1 / IoU
        return Fitn