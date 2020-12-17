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

Today = dt.datetime.now()
def Time_Forecasting():
    if DefFunc.month+DefFunc.year*12<=Today.month+Today.year*12:
        print("Start time must be less than end time!")
        exit()
    if DefFunc.number_in_stock<0:
        print("Number in stock must be positive!")
        exit()
    else:
        print("==========================================")
        print("	RUNNING SIMULATIONS	  	     ")
        print("==========================================")
        ts = time.time()    
        predicted_time = DefFunc.Estimated_time(DefFunc.unit_type,DefFunc.number_in_stock,DefFunc.month,DefFunc.year)
        te = time.time()
        print("==========================================")
        print("	SIMULATION PROCESS FINISHED!	  	 ")
        print("==========================================")
        if predicted_time ==0:
            print("The number of product actually in stock is %d, which can afford the need of customers."%DefFunc.number_in_stock)
        else:
            print("The number of product actually in stock is ",DefFunc.number_in_stock,", which can afford the need until ",predicted_time,".")
        print("Simulation time (by second): ", te-ts)
    
# To run tests without pytest (debug)
if __name__ == "__main__":
    Time_Forecasting()
