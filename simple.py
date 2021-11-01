# Copyright 2020-2021, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the Apache 2.0 License.

import gym
import numpy as np
import datetime
import pandas as pd

from simple_model_env import SimpleCropModelEnv

T_BASE = 0
LAT = 43.9271  # latitude of Hamer, ID; degrees N, used for ET0 computation
ELEV = 1464  # elevation of Hamer, ID in meters above sea level
WIND2 = 10.299802  # average wind speed in Hamer, ID in m/s (at 2 meter?)
VAP = 4.04  # vapor pressure in Hamer ID on 24 Apr 2021 based on a temp of 54F and dewpoint of 22, and according to
#   https://www.weather.gov/epz/wxcalc_vaporpressure
F_SOLAR_MAX = 0.95  # the maximum fraction of radiation interception that a crop can reach, governed by plant spacings,
#                       but typically set to 0.95 according to SIMPLE model paper:
#                       https://www.sciencedirect.com/science/article/pii/S1161030118304234#bib0205

class RandomWeatherSchedule:
    """should subclass from some type of WeatherSchedule object that validates settings?"""

    def __init__(self, num_days):
        # all made up data, loosely based on starting May 1st in Hamer, ID
        self.num_days = num_days
        rng = np.random.default_rng()
        max_temp = rng.normal(15, 3, size=num_days)  # degrees C
        min_temp = rng.normal(2, 2, size=num_days)  # degrees C
        mean_temp = rng.normal((max_temp - min_temp) / 2, 2, size=num_days)  # degrees C
        self.__dict__.update({
            'max_temp': max_temp,  # degrees C
            'min_temp': min_temp,  # degrees C
            'precipitation': np.maximum(0, rng.normal(0, 3, size=num_days)),  # mm
            'radiation': np.maximum(0, rng.normal(22, 1, size=num_days)),  # MJ/(m**2 * day)
            'co2': rng.normal(412, 1, size=num_days),  # PPM
            'avg_vapor_pressure': np.maximum(0, rng.normal(4.04, 1, size=num_days)),  # hPa
            'mean_temp': mean_temp,  # degrees C; not in SIMPLE model
            'avg_wind': np.maximum(0, rng.normal(8, 6, size=num_days))  # m/s; not in SIMPLE model
        })


class ConstantWeatherSchedule:
    """should subclass from some type of WeatherSchedule object that validates settings?"""

    def __init__(self, num_days):
        # all made up data, loosely based on starting May 1st in Hamer, ID
        self.num_days = num_days
        max_temp = np.ones(num_days) * 15  # degrees C
        min_temp = np.ones(num_days) * 2  # degrees C
        mean_temp = np.ones(num_days) * 7.5  # degrees C
        self.__dict__.update({
            'max_temp': max_temp,  # degrees C
            'min_temp': min_temp,  # degrees C
            'precipitation': np.ones(num_days) * 0.05,  # mm
            'radiation': np.ones(num_days) * 2,  # MJ/(m**2 * day)
            'co2': np.ones(num_days) * 412,  # PPM
            'avg_vapor_pressure': np.ones(num_days) * 4.04,  # hPa
            'mean_temp': mean_temp,  # degrees C; not in SIMPLE model
            'avg_wind': np.ones(num_days) * 8  # m/s; not in SIMPLE model
        })


class BentonWACSVWeatherSchedule:
    # todo: Define WeatherSchedule interface
    # Loads data from Benton E. station reading from https://weather.wsu.edu/?p=93050,
    # original version of data starts April 1st 2000 and goes 180 days
    def __init__(self, csv):
        db = pd.read_csv(csv)
        max_temp_key = db.columns[4]
        min_temp_key = db.columns[2]
        precip_key = db.columns[15]
        rad_key = db.columns[16]
        # co2 is nan for this csv
        dew_point_key = db.columns[5]
        mean_temp_key = db.columns[3]
        avg_wind_key = db.columns[9]
        self.__dict__.update({
            'max_temp': db[max_temp_key].to_numpy(),  # degrees C
            'min_temp': db[min_temp_key].to_numpy(),  # degrees C
            'precipitation': db[precip_key].to_numpy(),  # mm
            'radiation': db[rad_key].to_numpy(),  # MJ/(m**2 * day)
            'co2': np.ones(len(db)) * 1017,  # PPM,  not given in data, used average for Benton in 2021
            'avg_vapor_pressure': self.actual_vapor_pressure(db[dew_point_key].to_numpy()),  # hPa
            'mean_temp': db[mean_temp_key].to_numpy(),  # degrees C; not in SIMPLE model
            'avg_wind': db[avg_wind_key].to_numpy() * ((60 * 60) / 1000)  # convert from km/h to m/s; not in SIMPLE
        })


    @staticmethod
    def actual_vapor_pressure(dewpoint):
        return 6.11 * 10 ** (7.5 * dewpoint / (237.3 + dewpoint))

class WeatherForecastSTDs:
    # todo: Define WeatherForcastSTDs interface
    def __init__(self):
        self.__dict__.update({
            'max_temp': 3,  # degrees C
            'min_temp': 2,  # degrees C
            'precipitation': 3,  # mm
            'radiation': 1,  # MJ/(m**2 * day)
            'co2': 1,  # PPM
            'avg_vapor_pressure': 1,  # hPa
            'mean_temp': 2,  # degrees C; not in SIMPLE model
            'avg_wind': 6  # m/s; not in SIMPLE model
        })

class PotatoRussetUSACropParametersSpec:
    # Todo: Define CropParameterSpec interface
    """should subclass from some type of CropSpec object that validates settings?"""

    def __init__(self):
        # crop parameters
        self.temp_base = 4  # T_base
        self.temp_opt = 22  # T_opt
        self.RUE = 1.30  # RUE
        self.rad_50p_growth = 500  # I_50A
        self.rad_50p_senescence = 400  # I_50B
        self.maturity_temp = 2300  # T_sum
        self.rad_50p_max_heat = 50  # I_50maxH
        self.rad_50p_max_water = 30  # I_50maxW
        self.heat_stress_thresh = 34  # T_max
        self.heat_stress_extreme = 45  # T_ext
        self.drought_stress_sensitivity = 0.4  # S_water
        self.deep_drainage_coef = 0.8  # DDC
        self.water_holding_capacity = 0.1  # AWC
        self.runoff_curve_number = 64  # RCN
        self.root_zone_depth = 1200  # RZD
        self.co2_sensitivity = 0.10  # S_CO2
        self.harvest_index = 0.9  # HI


if __name__ == "__main__":
    growth_days = 180
    start_date = datetime.datetime(day=1, month=5, year=2021)
    # weather = ConstantWeatherSchedule(growth_days)  # RandomWeatherSchedule(growth_days)
    csv_path = 'data/daily_weather_ag_data_washstateu.csv'
    weather = BentonWACSVWeatherSchedule(csv_path)
    weather_stds = WeatherForecastSTDs()
    crop_params = PotatoRussetUSACropParametersSpec()
    env = SimpleCropModelEnv(start_date, growth_days, weather, WeatherForecastSTDs(), LAT, ELEV, crop_params, seed=0)
    env.reset()
    done = False
    iter = 0
    biomasses = []
    cum_temps = []
    arids = []
    f_solars = []
    paw = []
    cum_reward = 0
    while not done and iter < 1000000:
        action = [10]
        s, r, done, info = env.step(action)
        cum_reward += r
        print("state:", s)
        print(info['plant_available_water'])
        print("reward: ", r)
        paw.append(info['plant_available_water'])
        biomasses.append(info['cumulative_biomass'])
        cum_temps.append(info['cumulative_mean_temp'])
        arids.append(info['arid_index'])
        f_solars.append(info['f_solar'])
        iter += 1

    print("cumulative reward:", cum_reward)

    from matplotlib import pyplot as plt
    plt.plot(np.array(biomasses), label='biomass (t/hect)')
    plt.title('biomass')
    plt.figure()
    plt.plot(np.array(cum_temps), label='temp')
    plt.title('temp')
    plt.figure()
    plt.plot(np.array(arids), label='arid')
    plt.title('arid')
    plt.figure()
    plt.plot(np.array(paw), label='paw')
    plt.title('paw')
    plt.figure()
    plt.plot(np.array(f_solars), label='f_solar')
    plt.title('f_solar')
    plt.legend()
    plt.show()
