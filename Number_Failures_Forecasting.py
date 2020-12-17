import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm
import warnings
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines.utils import find_best_parametric_model
import datetime as dt
import time

import DefFunc

def Number_Failures_Forecasting():
    if DefFunc.alpha <= 0 or DefFunc.alpha >= 1:
        print("Confidence level must be in (0,1)!")
        exit(2)
    if DefFunc.month not in [1,2,3,4,5,6,7,8,9,10,11,12]:
        print("Invalid month!")
        exit(3)
    if DefFunc.year*12 + DefFunc.month < DefFunc.Today.year*12 + DefFunc.Today.month:
        print("Predicted day is in the pass!")
        exit(4)
    if DefFunc.company not in DefFunc.list_company and DefFunc.company!=0:
        print("Company name does not exist!")
        print("Company name must be integer of list: ", DefFunc.list_company, " or 0 if we want to forecast for ALL COMPANIES.")
        exit(5)
    if DefFunc.unit_type not in DefFunc.types:
        print("Type unit does not exist!")
        print("Unit type must be string of list: ", DefFunc.types)
        exit(6)
    print("==========================================")
    print("	RUNNING SIMULATIONS	  	     ")
    print("==========================================")
    ts = time.time()
    if DefFunc.company!=0:
        s,y,ci1,ci2,t = DefFunc.Estimated_Stock(DefFunc.company,DefFunc.unit_type,DefFunc.year,DefFunc.month)
    else:
        s,y,ci1,ci2,t = DefFunc.Estimated_Stock_All_Companies(DefFunc.unit_type,DefFunc.year,DefFunc.month,message=True)
    te = time.time()
    print("==========================================")
    print("	SIMULATION PROCESS FINISHED!	  	 ")
    print("==========================================")
    if DefFunc.company!=0:
        print("Forecast for type %s unit of company %d from %d/%d until %d/%d:"%(DefFunc.unit_type,DefFunc.company,DefFunc.Today.month,DefFunc.Today.year,DefFunc.month,DefFunc.year))
    else:
        print("Forecast for type %s unit from %d/%d until %d/%d:"%(DefFunc.unit_type,DefFunc.Today.month,DefFunc.Today.year,DefFunc.month,DefFunc.year))
    print("There are %d units which is actually on aircraft."%t)
    print("Predicting a number of unit in average for stock: ",s)
    print("with confidence interval (%0.2f,%0.2f) and confidence interval in average (%0.2f,%0.2f) at level %0.2f"%(ci1[0],ci1[1],ci2[0],ci2[1],100-100*DefFunc.alpha), end="")
    print("%.")
    print("Simulation time (by second): ", te-ts)
    
# To run tests without pytest (debug)
if __name__ == "__main__":
    Number_Failures_Forecasting()
