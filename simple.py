import numpy as np
from pcse.util import penman_monteith
import datetime
import gym

T_BASE = 0
LAT = 43.9271  # latitude of Hamer, ID; degrees N, used for ET0 computation
ELEV = 1464  # elevation of Hamer, ID in meters above sea level
WIND2 = 10.299802  # average wind speed in Hamer, ID in m/s (at 2 meter?)
VAP = 4.04  # vapor pressure in Hamer ID on 24 Apr 2021 based on a temp of 54F and dewpoint of 22, and according to
#   https://www.weather.gov/epz/wxcalc_vaporpressure
F_SOLAR_MAX = 0.95  # the maximum fraction of radiation interception that a crop can reach, governed by plant spacings,
#   but typically set to 0.95 according to SIMPLE model paper:
#   https://www.sciencedirect.com/science/article/pii/S1161030118304234#bib0205


def dq_d_temp(temp):
    """
    derivative of specific humidity/saturated vapour pressure at a given temperature
    See:
        - https://journals.ametsoc.org/view/journals/mwre/100/2/1520-0493_1972_100_0081_otaosh_2_3_co_2.xml?tab_body=pdf
        - A COMPARISONOF THE PRIESTLEY-TAYLOR AND PENMAN METHODS FOR ESTIMATING REFERENCE CROP EVAPOTRANSPIRATION IN
            TROPICAL COUNTRIES, H. GUNSTON and C.H. BATCHELOR
        - https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml

        P_s = a * e^( bt / (t + c) )
            where a = 610.94, b = 17.625, and c = 243.04

        so d(P_s)/dt = (b * t / (t + c)) * ((b * (t + c) - b * c* t) / ((t + c) ** 2) * a * e^( bt / (t + c) )
    """
    a, b, c = 610.94, 17.625, 243.04
    front = (b * temp / (temp + c)) * ((b * (temp + c) - b * c * temp) / ((temp + c) ** 2) * a)
    return front * np.exp(b * temp / (temp + c))


def calc_plant_available_water(paw_day_before, precipitation, irrigation, transpiration, deep_drainage_coef,
                               root_zone_depth, water_holding_capacity, runoff_curve_number):
    # T_i = min(alpha * zeta * theta^(ad)_{a, i-1}, ET0_i) = min(alpha * PAW_{i-1}, ET0_i)
    # ~ min(alpha * W, ET0_i)
    # W = W_{i-1} + P_i + I_i - T_i - D_i - R_i
    # P_i: precipitation on day i in mm
    # I_i: irrigation on day i in mm
    # T_i: transpiration
    # D_i: deep drainage
    # R_i: surface runoff
    # alpha = 0.096, beta = ddc, zeta = rzd, eta = 65 ;
    # https://acsess.onlinelibrary.wiley.com/doi/full/10.2134/agronj2011.0286
    plant_available_water = paw_day_before + precipitation + irrigation - transpiration
    deep_drainage = max(0, (deep_drainage_coef * root_zone_depth) * (plant_available_water - water_holding_capacity))
    potential_maximum_retention = 25400 / runoff_curve_number - 254
    initial_abstraction = 0.2 * potential_maximum_retention
    if precipitation > initial_abstraction:
        surface_runoff = ((precipitation - initial_abstraction) * (precipitation - initial_abstraction) /
                          (precipitation - initial_abstraction + potential_maximum_retention))
    else:
        surface_runoff = 0
    plant_available_water = plant_available_water - deep_drainage - surface_runoff
    return plant_available_water


def calc_transpiration(ref_evapotranspiration, plant_available_water):
    return min(0.096 * plant_available_water, ref_evapotranspiration)


def delta_cumulative_temp(temp, temp_base):
    return temp - temp_base if temp > temp_base else 0


def calc_biomass_rate(radiation, f_solar, rue, f_co2, f_temp, f_heat, f_water):
    return radiation * f_solar * rue * f_co2 * f_temp * min(f_heat, f_water)


def calc_cumulative_biomass(biomass_day_before, biomass_rate):
    return biomass_day_before + biomass_rate


def calc_yield(end_cumulative_biomass, harvest_index):
    return end_cumulative_biomass * harvest_index


def calc_f_solar(cumulative_mean_temp, rad_50p_growth, maturity_temp, rad_50p_senescence):
    # assumes maturity or senescence after cumulative mean temperature is grater than maturity_temp.
    if cumulative_mean_temp >= maturity_temp:
        return F_SOLAR_MAX / (1 + np.exp(0.01 * (cumulative_mean_temp - (maturity_temp - rad_50p_senescence))))
    else:
        return F_SOLAR_MAX / (1 + np.exp(-0.01 * (cumulative_mean_temp - rad_50p_growth)))


def calc_f_temp(temp, temp_base, temp_opt):
    if temp < temp_base:
        return 0
    elif temp < temp_opt:
        return (temp - temp_base) / (temp_opt - temp_base)
    else:
        return 1


def calc_f_heat(max_temp_day, crop_heat_stress_thresh, crop_extreme_heat_stress_thresh):
    if max_temp_day <= crop_heat_stress_thresh:
        return 1
    elif max_temp_day <= crop_extreme_heat_stress_thresh:
        return 1 - ((max_temp_day - crop_heat_stress_thresh)
                    / (crop_extreme_heat_stress_thresh - crop_heat_stress_thresh))
    else:
        return 0


def calc_f_co2(s_co2, co2):
    if 350 <= co2 <= 700:
        return 1 + s_co2 * (co2 - 350)
    elif co2 > 700:
        return 1 + s_co2 * 350
    else:
        raise ValueError("Invalid CO2 value: {}".format(co2))


def calc_f_water(s_water, arid):
    return 1 - s_water * arid


def calc_arid(transpiration, ref_evapotranspiration):
    return 1 - transpiration / ref_evapotranspiration


def calc_rad_50p_senescence(rad_50p_before, rad_50p_max_heat, rad_50p_max_water, f_heat, f_water):
    # I_{50B} term
    return rad_50p_before + rad_50p_max_heat * (1 - f_heat) + rad_50p_max_water * (1 - f_water)


def calc_f_solar_water(f_water):
    return 0.9 + f_water if f_water < 0.1 else 1.0


class SimpleCropModelEnv(gym.Env):
    def __init__(self):
        self.cumulative_mean_temp = 0
        self.cumulative_biomass = 0
        self.rad_50p_senescence = None
        self.plant_available_water = 0

        # crop parameters
        self.crop_temp_base = None

    def step(self, action):
        # exterior parameters needed
        day_mean_temp = 20

        self.cumulative_mean_temp += delta_cumulative_temp(day_mean_temp, self.crop_temp_base)

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    day = datetime.datetime.today()
    temp_min = 23
    temp_max = 25  # made up
    avg_rad = 4
    ET0 = penman_monteith(day, LAT, ELEV, temp_min, temp_max, avg_rad, VAP, WIND2)
    T_i = calc_transpiration(ET0, 1)
    # US potato constants
    awc = 0.1
    rcn = 64
    ddc = 0.8
    rzd = 1200
    paw = calc_plant_available_water(0, 5, 2, T_i, ddc, rzd, awc, rcn)
    ARID = calc_arid(T_i, ET0)

    print(ARID)
