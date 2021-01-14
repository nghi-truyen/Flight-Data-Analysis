<img align="left" src="https://github.com/nghitruyen/Flight_Data_Analysis/blob/main/images/Logo_INSAvilletoulouse-RVB.png"> <img align="right" width="240" height="160" src="https://github.com/nghitruyen/Flight_Data_Analysis/blob/main/images/logo-AirbusFHS.png">
<br />
<br />
<br />
<br />

# Flight Data Analysis - Airbus FHS

## *Forecast of aircraft parts failures and optimization of spare parts stock management*

### Task overview:

We work on “Aircraft component reliability”, which is a vital contributing factor for dispatch reliability and aircraft system reliability. The main purpose of this project is to utilize some statistical models in order to predict future failure of aircraft parts, and then to optimize the stock management.

This project is carried out in favour of two Airbus services, the airline and the FHS (Flight Hour Services). At airlines, inoperative units on airplanes are removed if it fails and then are replaced with serviceable units that are provided by their site stock. The Airbus FHS here plays two important roles: providing serviceable units for the site stock and repairing products for their customers (the OEMs).

<p align="center">
  <img src="https://github.com/nghitruyen/Flight_Data_Analysis/blob/main/images/AirbusFHS_activities.png" width="700" />
</p>

The challenges is to:

- reduce removals of parts,
    
- reduce repair costs and,
    
- reduce investments (stock optimization).
    
In the new situation due to the pandemic, the airlines have less aircraft flying and more contracts ending. Consequently, they need to fine-tune their strategy “repair or stock unserviceable” to optimize the stock management. So the failure forecast of aircraft part is a significant underlying contributing factor in the proposals of such strategy.

### Tutorial:

We first download the project and then, the `lifelines` library in Python is required to execute the codes of project:

`$ pip install lifelines`

The version number of each package is presented in `requirements.txt`.

The notebook file `/doc/FlightDataAnalysis.ipynb` includes all of codes for the simulation process as well as the graphs to visualize.

The `input.txt` file is to calibrate the parameters of simulations:

- `<Data file name>`: name of data file which must be string.

- `<Confidence level>`: confidence level coefficient which must be in (0,1).

- `<Company name>`: company that we want to forecast, if `company==0` so we want to forecast for all companies. This parameter is only useful for `Number_Failures_Forecasting()` function. 

- `<Unit type>`: part number we want to forecast for.

- `<Month>` and `<Year>`: the moment that we want to forecast in the future. This is not useful for `Time_Forecasting()` function.

- `<Number of type unit actually on stock>`: number of product of part number `<Unit type>` available on stock at the moment. This only useful for `Time_Forecasting()` function. 

- `<Service level>`: percentage to which customers' need should be met. Only useful for `Time_Forecasting()` function.

- `<Repair rate>`: percentage to which failed products are replaced by repaired products instead of replaced by new.

**Attention:** The data set format (clone name, sheet name, etc.) should be similar to the format of the data on `/DataSet/` and the data file should be named without spaces.

For estimating of the number of failures at a determined time in the future, we run `Number_Failures_Forecasting.py` file with the command:

`$ python Number_Failures_Forecasting.py`

For evaluating the failure number as a function of time, we run `Evolution_Number_Failures_Forecasting.py` file:

`$ python Evolution_Number_Failures_Forecasting.py`

For estimating of the moment that we can not immediately afford the need of customers (called out-of-stock date), we run `Time_Forecasting.py` file:

`$ python Time_Forecasting.py`

Finally, the `/output` directory contains simulation output results.
