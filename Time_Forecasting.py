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

def Time_Forecasting():
    if DefFunc.month+DefFunc.year*12<=DefFunc.Today.month+DefFunc.Today.year*12:
        print("Start time must be less than end time!")
        exit()
    if DefFunc.number_in_stock<0:
        print("Number of products on stock must be positive!")
        exit()
    else:
        print("==========================================")
        print("	RUNNING SIMULATIONS	  	     ")
        print("==========================================")
        DefFunc.different_or_not(DefFunc.unit_type,message=True)
        ts = time.time()    
        predicted_time,r = DefFunc.Estimated_time(DefFunc.unit_type,DefFunc.number_in_stock,DefFunc.month,DefFunc.year)
        te = time.time()
        print("==========================================")
        print("	SIMULATION PROCESS FINISHED!	  	 ")
        print("==========================================")
        if predicted_time == 0:
            print("The number of product of type %s actually on stock is %d, which can afford immediately %0.2f"%(DefFunc.unit_type,DefFunc.number_in_stock,100*DefFunc.service_level),end="")
            print("%",end="")
            print(" the need of customers until %d/%d"%(DefFunc.month,DefFunc.year))
        else:
            print("The number of product of type %s actually on stock is %d, which can only afford immediately %0.2f"%(DefFunc.unit_type,DefFunc.number_in_stock,100*DefFunc.service_level),end="")
            print("%",end="")
            print(" the need of customers until %d/%d"%(predicted_time.month,predicted_time.year))
            print("(","we are short of",round(r,2),"units to maintain this service level",").")
        print("Simulation time (by second): ", te-ts)
        return predicted_time,r,ts,te
    
# To run process and create output file
if __name__ == "__main__":
    predicted_time,r,ts,te = Time_Forecasting()
    save = "empty"
    while save not in ["y","n","yes","no"]:
        save = input("Do you want to save this result? (y/n)")
    if save in ["yes","y"]:
        name = "Time_Failure_at_"+str(DefFunc.Today)
        filename = "%s.out" % name
        filepath = os.path.join('./output', filename)
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f=open(filepath,"w+")
        
        f.write('INPUT'+ '\n')
        f.write("Unit type : " + DefFunc.unit_type + '\n')
        f.write("Forecasting for : " + str(DefFunc.month) + '/' + str(DefFunc.year) + '\n')
        f.write("Number of type unit actually on stock : "+str(DefFunc.number_in_stock)+'\n')
        f.write("Service level : " + str(DefFunc.service_level) + '\n')
        f.write("Repair rate : " + str(DefFunc.repair_rate) + '\n')
        f.write('\n')
        f.write("OUTPUT"+ '\n')
        f.write("Can products on stock afford immediately "+str(round(100*DefFunc.service_level,2))+"% the need of customers until that day? ")
        if predicted_time == 0:
            f.write("Yes"+'\n')
        else:
            f.write("No"+'\n')
            f.write("Estimated date the stock can no longer maintain this service level : " + str(predicted_time.month)+'/'+str(predicted_time.year)+'\n')
            f.write("Number of missing units to maintain service level : "+str(round(r,2))+'\n')
        f.write("Simulation time : " + str(round(te-ts,2))+' s'+'\n')
        f.close()
        print("The result is saved on ./output/",name)
