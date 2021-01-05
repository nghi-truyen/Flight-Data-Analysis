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

def Evolution_Number_Failures_Forecasting(gap):
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
    points,s=DefFunc.Time_series(DefFunc.unit_type,DefFunc.month,DefFunc.year,l=gap)
    te = time.time()
    print("==========================================")
    print("	SIMULATION PROCESS FINISHED!	  	 ")
    print("==========================================")
    print("Evolution of failure number of type %s from %d/%d until %d/%d:"%(DefFunc.unit_type,DefFunc.Today.month,DefFunc.Today.year,DefFunc.month,DefFunc.year))
    length=len(points)
    pts=[]
    for i in range(length):
        mois=int(points[i])%12+(int(points[i])%12==0)*12
        annee=int((points[i]-mois)/12)
        pts+=[str(dt.date(annee,mois,1))]
    print("Time : ", pts)
    print("Predicting failure number in average : ", s)
    print("With repair rate is", DefFunc.repair_rate)
    print("Simulation time (by second): ", te-ts)
    return pts,s,te,ts
    
# To run process and create output file
if __name__ == "__main__":
    while True:
        try:
            gap = int(input("A period time (by month) we want to evaluate for? (for example 1,2,3,...) : "))
            if gap<1:
                print("Oops! That was no valid number. Try again...")
            else:
                break
        except ValueError:
            print("Oops! That was no valid number. Try again...")
    points,s,te,ts = Evolution_Number_Failures_Forecasting(gap)
    save = "empty"
    while save not in ["y","n","yes","no"]:
        save = input("Do you want to save this result? (y/n)")
    if save in ["yes","y"]:
        name = "Evolution_Number_Failures_at_"+str(DefFunc.Today)
        filename = "%s.plt" % name
        filepath = os.path.join('./output', filename)
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f=open(filepath,"w+")
        
        f.write('INPUT'+ '\n')
        f.write("Unit type : " + DefFunc.unit_type + '\n')
        f.write("Period time by month : " + str(gap) + '\n')
        f.write("Forecasting for : " + str(DefFunc.month) + '/' + str(DefFunc.year) + '\n')
        f.write("Repair rate : " + str(DefFunc.repair_rate) + '\n')
        f.write('\n')
        f.write("OUTPUT"+ '\n')
        f.write("Time   "+"Number of failures in average" + '\n')
        for i in range(len(s)):
            f.write(str(points[i])+'     '+str(s[i])+'\n')
        f.write("Simulation time : " + str(round(te-ts,2))+' s'+'\n')
        
        f.close()
        print("The result is saved on ./output/",name)
