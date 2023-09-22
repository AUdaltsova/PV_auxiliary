import xarray as xr
import pandas as pd
import numpy as np

xr_train = xr.open_dataset(path + "df_15_train_full.netcdf")
xr_test = xr.open_dataset(path + "df_15_test_full.netcdf")

from datetime import datetime

def days_of_year_in_pi(datetime_array):
  days_of_year = np.array(list(map(lambda x: pd.Timestamp(x).timetuple().tm_yday, datetime_array)))
  res = ((days_of_year - 1)/365)*(2*np.pi)
  return res

def minutes_of_day_in_pi(datetime_array):
  minute_of_day = np.array(list(map(lambda x: (pd.Timestamp(x).to_pydatetime().hour * 60)+pd.Timestamp(x).to_pydatetime().minute, datetime_array)))
  res = (minute_of_day / (24*60))*(2*np.pi)
  return res

xr_train = xr_train.assign_coords({"date_sin": np.sin(days_of_year_in_pi(xr_train.datetime.data)),
                        "date_cos": np.cos(days_of_year_in_pi(xr_train.datetime.data)),
                        "time_sin": np.sin(minutes_of_day_in_pi(xr_train.datetime.data)),
                        "time_cos": np.cos(minutes_of_day_in_pi(xr_train.datetime.data))}).copy()

xr_test = xr_test.assign_coords({"date_sin": np.sin(days_of_year_in_pi(xr_test.datetime.data)),
                        "date_cos": np.cos(days_of_year_in_pi(xr_test.datetime.data)),
                        "time_sin": np.sin(minutes_of_day_in_pi(xr_test.datetime.data)),
                        "time_cos": np.cos(minutes_of_day_in_pi(xr_test.datetime.data))}).copy()

xr_train.to_netcdf(path+'xr_train_timestamped.netcdf')
xr_test.to_netcdf(path+'xr_test_timestamped.netcdf')
