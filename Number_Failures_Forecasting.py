import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm
import warnings
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines.utils import find_best_parametric_model
from lifelines.statistics import logrank_test
import datetime as dt
import time
import os

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
    DefFunc.different_or_not(DefFunc.unit_type,message=True)
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
    print("with Empirical Confidence Interval (%0.2f,%0.2f), CLT Confidence Interval (%0.2f,%0.2f) at level %0.2f"%(ci1[0],ci1[1],ci2[0],ci2[1],100-100*DefFunc.alpha), end="")
    print("% and with repair rate is", DefFunc.repair_rate)
    print("Simulation time (by second): ", te-ts)
    return s,y,ci1,ci2,t,te,ts
    
# To run process and create output file
if __name__ == "__main__":
    s,y,ci1,ci2,t,te,ts = Number_Failures_Forecasting()
    save = "empty"
    while save not in ["y","n","yes","no"]:
        save = input("Do you want to save this result? (y/n)")
    if save in ["yes","y"]:
        name = "Number_Failures_at_"+str(DefFunc.Today).replace(" ","_").replace(":","-").replace(".","-")
        filename = "%s.out" % name
        filepath = os.path.join('./output', filename)
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f=open(filepath,"w+")
        
        f.write('INPUT'+ '\n')
        f.write("Unit type : " + DefFunc.unit_type + '\n')
        if DefFunc.company==0:
            comp = "All"
        else:
            comp = str(DefFunc.company)
        f.write('Company : '+comp+'\n')
        f.write("Forecasting for : " + str(DefFunc.month) + '/' + str(DefFunc.year) + '\n')
        f.write("Repair rate : " + str(DefFunc.repair_rate) + '\n')
        f.write('\n')
        f.write("OUTPUT"+ '\n')
        f.write("Units which is actually on aircraft : " + str(t) + '\n')
        f.write("Number of failures in 1000 simulation times : " + str(y)+'\n')
        f.write("Number of failures in average : "+str(round(s,1))+'\n')
        f.write("Empirical Confidence Interval at "+str(round(100-100*DefFunc.alpha,2))+"% : "+'('+str(round(ci1[0],2))+','+str(round(ci1[1],2))+')'+'\n')
        f.write("Simulation time : " + str(round(te-ts,2))+' s'+'\n')
        
        f.close()
        print("The result is saved on ./output/",name)
