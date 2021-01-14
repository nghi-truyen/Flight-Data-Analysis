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

##############################################
##	INPUT PARAMETERS CALIBRATION        ##
##############################################

import_by_txt_file = True 	# TRUE IF WE WANT TO IMPORT INPUT PARAMETERS BY .TXT FILE.

if import_by_txt_file:
    # Import input.txt file:
    fname = open("input.txt","r")
    input_ = fname.readlines()
    fname.close()
    # Set up parameters
    alpha = 1-round(float(input_[4]),2) # related by confidence level
    file_name = str(input_[1][:-1].replace(" ", ""))
    file_location = "./DataSet/"+file_name
    company = int(input_[7])         # must be in INTEGER: only useful for Number_Failures_Forecasting function
    # If company==0 so we want to forecast for ALL COMPANIES
    unit_type = str(input_[10][:-1].replace(" ", "")) # STRING
    year = int(input_[16])         # INTEGER This is not useful for for `Time_Forecasting()` function.
    month = int(input_[13])           # INTEGER This is not useful for for `Time_Forecasting()` function.
    number_in_stock = int(input_[19]) # (INTEGER): only useful for "Time_Forecasting()" function 
    service_level = float(input_[22]) # must be in (0,1): only useful for "Time_Forecasting()" function
    repair_rate = round(float(input_[25]),2) # must be in (0,1)
else:
    confidence_level = 0.95    # TO CALIBRATE (must be in (0,1))
    alpha = 1-confidence_level
    
    file_name = "Dataset_ter.xlsx" # TO CALIBRATE (must be STRING and finished by .xlsx)
    file_location = "./DataSet/"+file_name
    
    company = 5         # TO CALIBRATE (must be in INTEGER): only useful for "Number_Failures_Forecasting()" function
    # !!! If company==0 so we want to forecast for ALL COMPANIES
    unit_type = 'A' # TO CALIBRATE (STRING)
    year = 2022         # TO CALIBRATE (INTEGER) This is not useful for for `Time_Forecasting()` function.
    month = 12           # TO CALIBRATE (INTEGER) This is not useful for for `Time_Forecasting()` function.
    number_in_stock = 60 # TO CALIBRATE (INTEGER): only useful for "Time_Forecasting()" function
    service_level = 0.9 # TO CALIBRATE (must be in (0,1)): only useful for "Time_Forecasting()" function
    repair_rate = 0.9 # must be in (0,1)

###############################################
##					     ##
###############################################

# Data preprocessing
def preprocessing(file_location):
    # Import data
    Removals = pd.read_excel(file_location, sheet_name='Removals')
    SNlist = pd.read_excel(file_location, sheet_name='SN list')
    airlines = pd.read_excel(file_location, sheet_name='Airlines')

    # Combining Removals and SNlist :
    fail_and_not = SNlist.copy()
    fail_and_not['On_Aircraft'] = False
    onaircraft_fan = fail_and_not['On_Aircraft']
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        onaircraft_fan[fail_and_not['Current SN Status Description']=='On Aircraft']=True
    fail_and_not['On_Aircraft'] = onaircraft_fan
    fail_and_not['failed'] = False
    failed_fan = fail_and_not['failed']
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        failed_fan[fail_and_not['Current SN Status Description']=='In Outside Repair']=True
    fail_and_not['failed'] = failed_fan
    fail_and_not = fail_and_not.drop(['Description','Current SN Status Description','Since New Date'], axis = 1)
    fail_and_not = fail_and_not.rename(columns={"Part Number": "PN", "Serial Number": "SN", "Hour ageing Since Installation": "TSI", "Hour ageing Since New": "TSN"})

    fail = Removals.copy()
    fail['On_Aircraft'] = False
    fail['failed'] = True
    failed_f = fail['failed']
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        failed_f[fail['Maintenance Type']=='Scheduled'] = False
    fail['failed'] = failed_f
    fail = fail.drop(['Removal date','Description','Maintenance Type'], axis=1)
    fail = fail.rename(columns={"P/N": "PN", "S/N": "SN", "TSI (Flight Hours) at removal": "TSI", "TSN (Flight Hours) at Removal": "TSN", "Customer":"Company"})

    all_SN = pd.unique(fail_and_not['SN'])
    SN_Removals = pd.unique(fail['SN'])

    combined = pd.concat([fail,fail_and_not], ignore_index=True)
    combined = combined.drop_duplicates(subset=['SN','PN','TSN'], keep='last')

    # Data errors treatment
    combined['TSI']=combined['TSI'].replace(np.nan, 0.0)
    combined['TSN']=combined['TSN'].replace(np.nan, 0.0)
    combined['Company']=combined['Company'].replace('1', 1)
    combined['Company']=combined['Company'].replace('3', 3)
    combined = combined[combined['TSN']!=0]
    return combined, airlines

def time_sticker(data_type): 
    T = data_type.TSI.to_numpy(dtype="float")
    d = np.array([1 if f == True else 0 for f in data_type.failed])
    return T,d

try: 
    print("==========================================")
    print("	DATA PREPROCESSING ...  	     ")
    print("==========================================")  
    combined, airlines = preprocessing(file_location)
    print("==========================================")
    print("	DATA PREPROCESSING FINISHED!	     ")
    print("==========================================") 
except FileNotFoundError:
    print("Data file does not exist!")
    exit(1)
    
types = pd.unique(combined["PN"])
types = types[np.logical_not(pd.isnull(types))]
data = combined.copy()
data_types = {}
for typ in types:
    data_types[typ] = data[data['PN']==typ]
    
def NAF(typ,df=data_types): # Nelson_Aalen model
    data_type = df[typ]
    T,d=time_sticker(data_type)
    label = "NA-estimator of type " + typ
    return NelsonAalenFitter().fit(T,d,alpha=alpha,label=label)

def KMF(typ,df=data_types): # Kaplan_meier model
    data_type = df[typ]
    T,d=time_sticker(data_type)
    label = "KM-estimator of type " + typ
    return KaplanMeierFitter().fit(T,d,alpha=alpha,label=label)
    
def best_parametric_model(typ,df=data_types): # find the best parametric model which bases in AIC (or BIC as we can change scoring_method="BIC") method.
    data_type = df[typ]
    T,d=time_sticker(data_type)
    T[T==0]=1e-6 # Avoid divising by zero.
    tau=1-sum(d)/len(d)
    if tau>0.9:
        warnings.warn("There are more 90% censored data in type {} data. The applied model might not be correct!".format(str(typ)))
    best_model = find_best_parametric_model(T, d, scoring_method="AIC")[0]
    return best_model
    
def old_vs_new(typ,df=data_types):
    data_type = df[typ]
    new_or_not = data_type["TSI"]==data_type["TSN"]
    old = data_type[new_or_not==False]
    new = data_type[new_or_not==True]
    results = logrank_test(old["TSI"], new["TSI"], event_observed_A=old["failed"], event_observed_B=new["failed"])
    return results,(len(old.to_numpy()), len(new.to_numpy()))
    
def inverse_sampling(kapmei, timeline):
    u = np.random.uniform()
    if u < kapmei[-1]:
        T = -1
    elif u > kapmei[0]:
        T = 0
    else:
        arg = np.argmax(kapmei<=u)-1
        T = float(timeline[arg]+(timeline[arg+1]-timeline[arg])*(kapmei[arg]-u)/(kapmei[arg]-kapmei[arg+1]))
    return T

def conditional_inverse_sampling(kapmei, timeline, TSI):
    T = 0
    while T<=TSI and T>=0:
        T = inverse_sampling(kapmei, timeline)
    return T-TSI

def num_of_fails_indivi_kapmei(TSI, T, kapmei, timeline):    
    t = conditional_inverse_sampling(kapmei, timeline, TSI)
    if t <= T:
        n_fails = 0 
        sum_t = (t<0)*np.max(timeline) + (t>=0)*t
        while sum_t <= T:
            t = inverse_sampling(kapmei, timeline)
            sum_t += (t<0)*np.max(timeline) + (t>=0)*t
            n_fails += 1
        return n_fails
    else: 
        return 0

def num_of_fails_list(TSI_list, T, kapmei, timeline):
    n_fails_list = []
    for TSI in TSI_list:
        n_fails = num_of_fails_indivi_kapmei(TSI, T, kapmei, timeline)
        n_fails_list += [n_fails]
    total_fails = np.sum(n_fails_list)
    return total_fails
    
def num_of_fails_indivi_kapmei_diff(TSI, TSN,T,k_old,t_old,k_new,t_new,rate):
    if TSI==TSN:
        t = conditional_inverse_sampling(k_new, t_new, TSI)
        cum = (t<0)*np.max(t_new)
    else:
        t = conditional_inverse_sampling(k_old, t_old, TSI)
        cum = (t<0)*np.max(t_old)
    if t <= T:
        n_fails = 0 
        sum_t = cum + (t>=0)*t
        while sum_t <= T:
            if np.random.uniform(0,1)<rate:
                t = inverse_sampling(k_old, t_old)
                sum_t += (t<0)*np.max(t_old) + (t>=0)*t
            else:
                t = inverse_sampling(k_new, t_new)
                sum_t += (t<0)*np.max(t_new) + (t>=0)*t
            n_fails += 1
        return n_fails
    else: 
        return 0

def num_of_fails_list_diff(TSI_list, TSN_list,T,k_old,t_old,k_new,t_new,rate):
    n_fails_list = []
    for i in range(len(TSI_list)):
        n_fails = num_of_fails_indivi_kapmei_diff(TSI_list[i],TSN_list[i],T,k_old,t_old,k_new,t_new,rate)
        n_fails_list += [n_fails]
    total_fails = np.sum(n_fails_list)
    return total_fails
    
# Confidence Intervals
def CI(Y,alpha=alpha):
    n=len(Y)
    # Confidence Interval of samples
    low1=np.quantile(Y,alpha/2)
    high1=np.quantile(Y,1-alpha/2)
    # Confidence Interval of mean (by Central Limit Theorem)
    mu=np.mean(Y)
    std=np.std(Y)
    q=norm.ppf(1-alpha/2, loc=0, scale=1)
    S=std*q/np.sqrt(n)
    low2=np.max([0.0,mu-S]) 
    high2=mu+S
    return (low1,high1),(low2,high2)

list_company = pd.unique(airlines["Company"])
list_company = list_company[np.logical_not(pd.isnull(list_company))]
Today = dt.datetime.now()

def different_or_not(typ,df=data_types,message=False):
    r,old_new=old_vs_new(typ,df=df)
    pval=r.p_value
    diff=(pval<0.05) and (min(old_new)/sum(old_new)>=0.1)
    if diff:
        if message:
            print("The old parts and new parts of type %s have different distributions!"%typ)
    return diff

def Estimated_Stock(company,typ,year,month,df=data,df_types=data_types,airlines=airlines,Begin=Today,MC=1000,rate=repair_rate):
# MC is number iteration of Monte-Carlo
    
    FH_per_month = float(airlines[airlines['Company']==company]['FH per aircraft per month'])
    End = dt.datetime(year, month, 1)
    FH_till_end = FH_per_month*((End.year-Begin.year)*12+End.month-Begin.month)
    diff = different_or_not(typ,df=df_types)
    
    if diff:
        df_types_diff = df_types.copy()
        df_types_diff.pop(typ,None)
        df_types_diff['new']=df_types[typ][df_types[typ].TSI==df_types[typ].TSN]
        df_types_diff['old']=df_types[typ][df_types[typ].TSI!=df_types[typ].TSN]
        kmf_old = KMF("old",df=df_types_diff)
        kmf_new = KMF("new",df=df_types_diff)
        surv_old = kmf_old.survival_function_.to_numpy()
        time_old = kmf_old.timeline
        surv_new = kmf_new.survival_function_.to_numpy()
        time_new = kmf_new.timeline
        if FH_till_end>np.max(time_old): 
            warnings.warn("Kaplan-Meier model of the old part of type %s can not estimate the stock until %d/%d. We apply the best parametric model to predict in this case."%(typ,month,year))
            bpm_old=best_parametric_model("old",df=df_types_diff)
            print("The best parametric model applied for old parts is:",bpm_old)
            time_old = np.linspace(0,FH_till_end,2000)
            surv_old = bpm_old.survival_function_at_times(time_old).to_numpy()

        if FH_till_end>np.max(time_new):
            warnings.warn("Kaplan-Meier model of the new part of type %s can not estimate the stock until %d/%d. We apply the best parametric model to predict in this case."%(typ,month,year))
            bpm_new=best_parametric_model("new",df=df_types_diff)
            print("The best parametric model applied for new parts is:",bpm_new)
            time_new = np.linspace(0,FH_till_end,2000)
            surv_new = bpm_new.survival_function_at_times(time_new).to_numpy()
    else:
        kmf=KMF(typ,df=df_types)
        survival = kmf.survival_function_.to_numpy()
        timeline = kmf.timeline
        if FH_till_end>np.max(timeline):
            warnings.warn("Kaplan-Meier model of type %s can not estimate the stock until %d/%d. We apply the best parametric model to predict in this case."%(typ,month,year))
            bpm = best_parametric_model(typ,df=df_types)
            print("The best parametric model applied for this type is:",bpm)
            timeline = np.linspace(0,FH_till_end,2000)
            survival = bpm.survival_function_at_times(timeline).to_numpy()

    dat = df[df.Company==company]
    dat = dat[dat.PN==typ]
    dat = dat[dat.On_Aircraft==True]
    total = len(dat.TSI)
  
    list_TSI = dat[dat.failed==False].TSI.to_numpy()
    list_TSN = dat[dat.failed==False].TSN.to_numpy()

    stock = 0
    y=[]
    for i in range(MC):
        if diff:
            a = num_of_fails_list_diff(list_TSI,list_TSN,FH_till_end,surv_old,time_old,surv_new,time_new,rate)
        else:
            a = num_of_fails_list(list_TSI,FH_till_end,survival,timeline)
        y += [a]
        stock += a
    stock = stock/MC  
    ## CI
    ci1,ci2=CI(y)
    return stock,y,ci1,ci2,total
    
def Estimated_Stock_All_Companies(typ,year,month,df=data,df_types=data_types,airlines=airlines,Begin=Today,MC=1000,message=False,rate=repair_rate):
    s,y,ci1,ci2,t = 0,np.zeros(MC),np.zeros(2),np.zeros(2),0
    for company in list_company:
        mm,yyyy=airlines['End of contract'][company-1].month,airlines['End of contract'][company-1].year
        if mm+yyyy*12<month+year*12:
            s_,y_,ci1_,ci2_,t_ = Estimated_Stock(company,typ,yyyy,mm,df=df,df_types=df_types,airlines=airlines,Begin=Begin,MC=MC,rate=rate)
            if message:
                print('The contract of company %d will end before %d/%d (in %d/%d)'%(company,month,year,mm,yyyy))
        else:
            s_,y_,ci1_,ci2_,t_ = Estimated_Stock(company,typ,year,month,df=df,df_types=df_types,airlines=airlines,Begin=Begin,MC=MC,rate=rate)
        s += s_
        y += y_
        ci1 += np.array(ci1_)
        ci2 += np.array(ci2_)
        t += t_
    return s,y,ci1,ci2,t
    
def Time_series(typ,month,year,df=data,df_types=data_types,airlines=airlines,Begin=Today,MC=1000,l=6):
    start = Begin.month+Begin.year*12
    end = month+year*12
    if end<=start:
        print("Start time must be less than end time!")
        exit()
    else:
        points = np.linspace(start,end,end-start+1)
        gap=len(points)-1
        ind=[l*i for i in range(int(gap/l)+1)]+[-1]*int(gap%l!=0)
        points=np.array([points[i] for i in ind]) # assuming that number of faillures is linear for every l=6 month
        s = np.zeros(len(points))
        s_lower = np.zeros(len(points))
        s_upper = np.zeros(len(points))
        for i in range(len(points)):
            mois=int(points[i])%12+(int(points[i])%12==0)*12
            ss,w,ci,r,q = Estimated_Stock_All_Companies(typ,int((points[i]-mois)/12),mois,df=df,df_types=df_types,airlines=airlines,Begin=Begin,MC=MC)
            s[i]=ss
            s_lower[i] = ci[0]
            s_upper[i] = ci[1]
    return points,s,s_lower,s_upper

def lifetime_simulation_indivi(TSI, T, kapmei, timeline):    
    t = conditional_inverse_sampling(kapmei, timeline, TSI)
    serie = []
    if t <= T: 
        sum_t = (t<0)*np.max(timeline) + (t>=0)*t
        while sum_t <= T:
            serie += [sum_t]
            t = inverse_sampling(kapmei, timeline)
            sum_t += (t<0)*np.max(timeline) + (t>=0)*t
    return serie

def lifetime_simulation_list(TSI_list, T, kapmei, timeline):
    series = []
    for TSI in TSI_list:
        serie = lifetime_simulation_indivi(TSI, T, kapmei, timeline)
        series += serie
    return np.sort(series)
    
def lifetime_simulation_indivi_diff(TSI, TSN,T,k_old,t_old,k_new,t_new,rate):
    if TSI==TSN:
        t = conditional_inverse_sampling(k_new, t_new, TSI)
        cum = (t<0)*np.max(t_new)
    else:
        t = conditional_inverse_sampling(k_old, t_old, TSI)
        cum = (t<0)*np.max(t_old)
    serie = []
    if t <= T:
        sum_t = cum + (t>=0)*t
        while sum_t <= T:
            serie += [sum_t]
            if np.random.uniform(0,1)<rate:
                t = inverse_sampling(k_old, t_old)
                sum_t += (t<0)*np.max(t_old) + (t>=0)*t
            else:
                t = inverse_sampling(k_new, t_new)
                sum_t += (t<0)*np.max(t_new) + (t>=0)*t
    return serie

def lifetime_simulation_list_diff(TSI_list, TSN_list,T,k_old,t_old,k_new,t_new,rate):
    series = []
    for i in range(len(TSI_list)):
        serie = lifetime_simulation_indivi_diff(TSI_list[i],TSN_list[i],T,k_old,t_old,k_new,t_new,rate)
        series += serie
    return np.sort(series)
    
def Estimated_time_company(company,typ,year,month,df=data,df_types=data_types,airlines=airlines,Begin=Today,MC=1000,rate=repair_rate,tau=service_level):
    FH_per_month = float(airlines[airlines['Company']==company]['FH per aircraft per month'])
    End = dt.datetime(year, month, 1)
    FH_till_end = FH_per_month*((End.year-Begin.year)*12+End.month-Begin.month)
    diff = different_or_not(typ,df=df_types)

    if diff:
        df_types_diff = df_types.copy()
        df_types_diff.pop(typ,None)
        df_types_diff['new']=df_types[typ][df_types[typ].TSI==df_types[typ].TSN]
        df_types_diff['old']=df_types[typ][df_types[typ].TSI!=df_types[typ].TSN]
        kmf_old = KMF("old",df=df_types_diff)
        kmf_new = KMF("new",df=df_types_diff)
        surv_old = kmf_old.survival_function_.to_numpy()
        time_old = kmf_old.timeline
        surv_new = kmf_new.survival_function_.to_numpy()
        time_new = kmf_new.timeline
        if FH_till_end>np.max(time_old): 
            warnings.warn("Kaplan-Meier model of the old part of type %s can not estimate the stock until %d/%d. We apply the best parametric model to predict in this case."%(typ,month,year))
            bpm_old=best_parametric_model("old",df=df_types_diff)
            print("The best parametric model applied for old parts is:",bpm_old)
            time_old = np.linspace(0,FH_till_end,2000)
            surv_old = bpm_old.survival_function_at_times(time_old).to_numpy()
        if FH_till_end>np.max(time_new):
            warnings.warn("Kaplan-Meier model of the new part of type %s can not estimate the stock until %d/%d. We apply the best parametric model to predict in this case."%(typ,month,year))
            bpm_new=best_parametric_model("new",df=df_types_diff)
            print("The best parametric model applied for new parts is:",bpm_new)
            time_new = np.linspace(0,FH_till_end,2000)
            surv_new = bpm_new.survival_function_at_times(time_new).to_numpy()
    else:
        kmf=KMF(typ,df=df_types)
        survival = kmf.survival_function_.to_numpy()
        timeline = kmf.timeline
        if FH_till_end>np.max(timeline):
            warnings.warn("Kaplan-Meier model of type %s can not estimate the stock until %d/%d. We apply the best parametric model to predict in this case."%(typ,month,year))
            bpm = best_parametric_model(typ,df=df_types)
            print("The best parametric model applied for this type is:",bpm)
            timeline = np.linspace(0,FH_till_end,2000)
            survival = bpm.survival_function_at_times(timeline).to_numpy()

    dat = df[df.Company==company]
    dat = dat[dat.PN==typ]
    dat = dat[dat.On_Aircraft==True]
    total = len(dat.TSI)
  
    list_TSI = dat[dat.failed==False].TSI.to_numpy()
    list_TSN = dat[dat.failed==False].TSN.to_numpy()
    
    times=[]
    for i in range(MC):
        if diff:
            time = lifetime_simulation_list_diff(list_TSI,list_TSN,FH_till_end,surv_old,time_old,surv_new,time_new,rate)
        else:
            time = lifetime_simulation_list(list_TSI,FH_till_end,survival,timeline)
        time = time/FH_per_month+12*Begin.year+Begin.month
        times += [time]
    return times

def Estimated_time(typ,N,df=data,df_types=data_types,airlines=airlines,Begin=Today,MC=1000,rate=repair_rate,tau=service_level,message=False):
    P = int(N/tau)
    total_times = []
    t_max=[]
    for company in list_company:
        mm,yyyy=airlines['End of contract'][company-1].month,airlines['End of contract'][company-1].year
        t_max+=[mm+yyyy*12]
        times = Estimated_time_company(company,typ,yyyy,mm,df=df,df_types=df_types,airlines=airlines,Begin=Begin,MC=MC,rate=rate)
        if message:
            print('The contract of company %d will end in %d/%d'%(company,mm,yyyy))
        total_times += [times]
    tmax=np.max(t_max)
    y=[]
    for i in range(MC):
        y_i = np.array([])
        for j in range(len(list_company)):
            y_i = np.append(y_i,total_times[j][i])
        y+=[np.sort(y_i)]
    time_estimated = np.array([y[i][P-1] if P<=len(y[i]) else tmax for i in range(MC)])
    ac = np.array([1 if time_estimated[i]==tmax else 0 for i in range(MC)])
    num_afford = np.sum(ac)
    if num_afford > 0:
        chance = num_afford/MC
        time_mean = chance
        ci = (np.quantile(time_estimated,alpha),tmax)
    else:
        time_mean = np.mean(time_estimated)
        ci,ci2 = CI(time_estimated)
    return time_mean,time_estimated,ci
    
def month_to_datetime(n):
    year=int(n/12)-1*(int(n)%12==0)
    m = n-year*12
    m = m*365/360  # correction for the assumption of month = 30 days
    delta=timedelta(days=m*30)
    return dt.date(year-1,12,1)+delta
