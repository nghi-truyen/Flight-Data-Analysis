<img align="left" src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png"> <img align="right" width="240" height="160" src="https://github.com/nghitruyen/Flight_Data_Analysis/blob/main/images/logo-AirbusFHS.png">
<br />
<br />
<br />
<br />

# Flight Data Analysis - Airbus FHS

## *Forecast of aircraft parts failures and optimization of spare parts stock management*

### Introduction:

We work on “Aircraft component reliability”, which is a vital contributing factor for dispatch reliability and aircraft system reliability. The main purpose of this project is to utilize the statistical models in order to predict if the parts of aircraft will be failure, and then to optimize the stock management.

We consider two services of Airbus, the airline and the FHS (Flight Hour Services). The Airbus planes are removed unserviceable unit if it failures and then are installed serviceable unit that is provided by their site stock. The Airbus FHS here plays two important roles: providing serviceable units for the site stock and repairing product for their customers (the OEMs).

<p align="center">
  <img src="https://github.com/nghitruyen/Flight_Data_Analysis/blob/main/images/AirbusFHS_activities.png" width="700" />
</p>

The challenges is to:

- reduce removals of parts,
    
- reduce repair costs and,
    
- reduce investments (stock optimization).
    
In the situation due to the pandemic, the airline has less aircraft flying and more contracts ending. Consequently, they need to fine-tune their strategy “repair or stock unserviceable” to optimize the stock management. So the fact of forecasting for the failure of aircraft part is a significant underlying contributing factor for the proposals of such strategy. 

### Tutorial:

We first download the project and then, the `lifelines` library in Python is required to execute the codes of project:

`$ pip install lifelines`

The notebook file includes all of codes for the simulation process as well as the graphs to visualize.

The `input.txt` file is to calibrate the parameters of simulations:

- `<Data file name>`: name of data file which must be string.

- `<Confidence level>`: confidence level coefficient which must be in (0,1).

- `<Company name>`: company that we want to forecast, if `company==0` so we want to forecast for all companies. This parameter is only useful for `Number_Failures_Forecasting()` function. 

- `<Unit type>`: type of unit we want to forecast for.

- `<Month>` and `<Year>`: the moment that we want to forecast in the future.

- `<Number of type unit actually in stock>`: number of product of type `<Unit type>` available on stock at the moment. This only useful for `Time_Forecasting()` function. 

- `<Service level>`: percentage to which customers' need shoud be met. Only useful for `Time_Forecasting()` function.

For estimating of the number of failures at a determined time in the future, we run `Number_Failures_Forecasting.py` file with the command:

`$ python Number_Failures_Forecasting.py`

For estimating of the moment that we can not afford the need of customers, we run `Time_Forecasting.py` file:

`$ python Time_Forecasting.py`
