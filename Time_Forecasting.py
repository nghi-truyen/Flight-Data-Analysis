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
from datetime import timedelta
import os

import DefFunc

def Time_Forecasting():
    if DefFunc.number_in_stock<0:
        print("Number of products on stock must be positive!")
        exit()
    else:
        print("==========================================")
        print("	RUNNING SIMULATIONS	  	     ")
        print("==========================================")
        DefFunc.different_or_not(DefFunc.unit_type,message=True)
        ts = time.time()    
        predicted_time,time_serie,ci1 = DefFunc.Estimated_time(DefFunc.unit_type,DefFunc.number_in_stock,message=True)
        te = time.time()
        print("==========================================")
        print("	SIMULATION PROCESS FINISHED!	  	 ")
        print("==========================================")
        ci=list(ci1)
        ci[0]=DefFunc.month_to_datetime(ci1[0])
        ci[1]=DefFunc.month_to_datetime(ci1[1])
        if predicted_time<=1:
            print("There is ",predicted_time*100,"% chance that we are not out of stock till the last end of contract in ",ci[1])
            print("Estimated date after which we have ",100-round(100*DefFunc.alpha,2),"% chance of running out of stock : ",ci[0])
            HaveChance = True
        else:
            predicted_time=DefFunc.month_to_datetime(predicted_time)
            print("The number of product of type %s actually in stock is %d, which can immediately afford %0.2f"%(DefFunc.unit_type,DefFunc.number_in_stock,100*DefFunc.service_level),end="")
            print("%",end="")
            print(" the need of customers until ",predicted_time,end="")
            print(" with Empirical Confidence Interval at ",100-round(100*DefFunc.alpha,2),"% : [",ci[0],ci[1],']')
            HaveChance = False
        print("Simulation time (by second): ", te-ts)
        return predicted_time,ci,ts,te,HaveChance
    
# To run process and create output file
if __name__ == "__main__":
    predicted_time,ci,ts,te,chance = Time_Forecasting()
    save = "empty"
    while save not in ["y","n","yes","no"]:
        save = input("Do you want to save this result? (y/n)")
    if save in ["yes","y"]:
        name = "Time_Failure_at_"+str(DefFunc.Today).replace(" ","_").replace(":","-").replace(".","-")
        filename = "%s.out" % name
        filepath = os.path.join('./output', filename)
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f=open(filepath,"w+")
        
        f.write('INPUT'+ '\n')
        f.write("Unit type : " + DefFunc.unit_type + '\n')
        f.write("Number of part actually on stock : "+str(DefFunc.number_in_stock)+'\n')
        f.write("Service level : " + str(DefFunc.service_level) + '\n')
        f.write("Repair rate : " + str(DefFunc.repair_rate) + '\n')
        f.write('\n')
        f.write("OUTPUT"+ '\n')
        if chance:
            f.write("Chance that we are not out of stock till the last end of contract in "+str(ci[1])+" : "+str(predicted_time*100)+"%"+'\n')
            f.write("Estimated date after which we have "+str(100-round(100*DefFunc.alpha,2))+"% chance of running out of stock : "+str(ci[0])+'\n')
        else:
            f.write("Estimated out-of-stock date with "+str(round(100*DefFunc.service_level,2))+"% service level : ")
            f.write(str(predicted_time)+'\n')
            f.write("Empirical Confidence Interval at "+str(100-round(100*DefFunc.alpha,2))+'% : ['+str(ci[0])+'  '+str(ci[1])+']'+'\n')
        f.write("Simulation time : " + str(round(te-ts,2))+' s'+'\n')
        f.close()
        print("The result is saved on ./output/",name)
